import torch
import torch.nn as nn
import torch.optim as optim
from time import time
import argparse
import numpy as np
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

# Détection du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Définition du modèle en PyTorch
class MLPModel(nn.Module):
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8], reg_layers=[0, 0, 0, 0]):
        super(MLPModel, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)
        
        # Fully connected layers
        layer_sizes = [layers[0]] + layers
        self.fc_layers = nn.ModuleList()
        
        for i in range(1, len(layer_sizes)):
            self.fc_layers.append(
                nn.Linear(layer_sizes[i-1], layer_sizes[i])
            )
            self.fc_layers.append(nn.ReLU())
        
        # Final prediction layer
        self.prediction = nn.Linear(layers[-1], 1)  # Assurez-vous que cela correspond au nombre de classes

    def forward(self, user_input, item_input):
        user_latent = self.user_embedding(user_input)
        item_latent = self.item_embedding(item_input)
        
        vector = torch.cat([user_latent, item_latent], dim=-1)
        
        for layer in self.fc_layers:
            vector = layer(vector)
        
        prediction = self.prediction(vector)
        
        return prediction
    
    def predict(self, user_input, item_input, batch_size=256):
        self.eval()  # Met le modèle en mode évaluation
        all_predictions = []

        # Prédictions par lots
        for i in range(0, len(user_input), batch_size):
            batch_user_input = torch.tensor(user_input[i:i + batch_size], dtype=torch.long).to(device)
            batch_item_input = torch.tensor(item_input[i:i + batch_size], dtype=torch.long).to(device)

            with torch.no_grad():  # Désactive la rétropropagation
                batch_preds = self.forward(batch_user_input, batch_item_input)
                all_predictions.append(batch_preds)

        return torch.cat(all_predictions, dim=0)

# Fonction pour générer des instances d'entraînement
def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_items = train.shape[1]
    for (u, i) in train.keys():
        # Positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)  # Positive instance label = 1
        
        # Negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)  # Negative instance label = 0
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
    
    topK = 10
    evaluation_threads = 1
    print("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%d.pth' % (args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d" 
          %(time()-t1, num_users, num_items, train.nnz, len(testRatings)))
    
    # Build model and move it to GPU if available
    model = MLPModel(num_users, num_items, layers, reg_layers).to(device)
    
    # Set optimizer
    if learner.lower() == "adagrad": 
        optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    elif learner.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif learner.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)    
    
    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        
        # Convert to tensors and move to GPU
        user_input = torch.LongTensor(user_input).to(device)
        item_input = torch.LongTensor(item_input).to(device)

        labels = torch.FloatTensor(labels).to(device)  # Labels should be LongTensor for CrossEntropyLoss
    
        # Training        
        model.train()
        optimizer.zero_grad()
        predictions = model(user_input, item_input)
        
        # Compute loss
        labels = labels.unsqueeze(1)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        t2 = time()

        # Evaluation
        if epoch % verbose == 0:  # Utilisation de verbose pour afficher les résultats chaque X epochs
            model.eval()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
            hr, ndcg, loss_val = np.array(hits).mean(), np.array(ndcgs).mean(), loss.item()
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
                  % (epoch,  t2-t1, hr, ndcg, loss_val, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                # torch.save(model.state_dict(), model_out_file)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))
