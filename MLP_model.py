import numpy as np
import pandas as pd
from keras import backend as K
from keras import initializers
from keras.models import Model
from keras.layers import Embedding, Input, Dense, Flatten, concatenate
from keras.regularizers import l2
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from time import time
import argparse
from evaluate import evaluate_model
from load_dataset import Dataset




#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations.')
    return parser.parse_args()

def init_normal(shape, name=None):
    return initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)(shape)

def get_model(num_users, num_items, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0]):
    assert len(layers) == len(reg_layers)
    # Input layers
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # Embedding layers
    user_embedding = Embedding(input_dim=num_users, output_dim=layers[0] // 2, name='user_embedding',
                                embeddings_regularizer=l2(reg_layers[0]), input_length=1)(user_input)
    item_embedding = Embedding(input_dim=num_items, output_dim=layers[0] // 2, name='item_embedding',
                                embeddings_regularizer=l2(reg_layers[0]), input_length=1)(item_input)

    # Flatten embeddings
    user_latent = Flatten()(user_embedding)
    item_latent = Flatten()(item_embedding)

    # Concatenate embeddings
    vector = concatenate([user_latent, item_latent])

    # Fully connected layers
    for idx in range(1, len(layers)):
        vector = Dense(layers[idx], activation='relu', kernel_regularizer=l2(reg_layers[idx]), name=f'layer{idx}')(vector)

    # Final prediction layer
    prediction = Dense(5, activation='softmax', name='prediction')(vector)

    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [],[],[]
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(train[u,i])
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    
    topK = 6
    evaluation_threads = 12
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%d.h5' % (args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(learning_rate=learning_rate), loss='categorical_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(learning_rate=learning_rate), loss='categorical_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy')
    else:
        model.compile(optimizer=SGD(learning_rate=learning_rate), loss='categorical_crossentropy')    
    
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
    
        # Training        
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch %verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" %(model_out_file))
    
    