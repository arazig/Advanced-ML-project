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

class MLPModel(nn.Module):
    def __init__(self, num_users, num_items, layers=[64, 32, 16, 8]):
        super(MLPModel, self).__init__()
        
        # Embedding layers for users and items
        self.user_embedding = nn.Embedding(num_users, layers[0] // 2)
        self.item_embedding = nn.Embedding(num_items, layers[0] // 2)
        
        # Fully connected layers
        layer_sizes = [layers[0]] + layers
        self.fc_layers = nn.Sequential()
        for i in range(1, len(layer_sizes)):
            self.fc_layers.add_module(f"fc{i}", nn.Linear(layer_sizes[i - 1], layer_sizes[i]))
            self.fc_layers.add_module(f"relu{i}", nn.ReLU())
        
        # Final prediction layer
        self.prediction = nn.Linear(layers[-1], 1)  # Single output (logit) for binary classification

    def forward(self, user_input, item_input):
        # Generate user and item embeddings
        user_latent = self.user_embedding(user_input)
        item_latent = self.item_embedding(item_input)
        
        # Concatenate user and item embeddings
        vector = torch.cat([user_latent, item_latent], dim=-1)
        
        # Pass through the fully connected layers
        vector = self.fc_layers(vector)
        
        # Final prediction (logit)
        prediction = self.prediction(vector).squeeze(-1)  # Remove the last unnecessary dimension
        
        # Apply sigmoïde to output logits and convert to probability
        probability = torch.sigmoid(prediction)
        
        return probability

    def predict(self, user_input, item_input, batch_size=100):
        # Switch the model to evaluation mode
        self.eval()
        all_predictions = []

        # Make predictions in batches
        for i in range(0, len(user_input), batch_size):
            batch_user_input = torch.tensor(user_input[i:i + batch_size], dtype=torch.long).to(device)
            batch_item_input = torch.tensor(item_input[i:i + batch_size], dtype=torch.long).to(device)

            with torch.no_grad():  # Disable gradient computation
                predictions = self.forward(batch_user_input, batch_item_input)
                all_predictions.append(predictions)

        return torch.cat(all_predictions, dim=0)


def get_train_instances(train, num_negatives):
    """ Generate user-item training instances with negative sampling """
    user_input, item_input, labels = [], [], []
    num_users, num_items = train.shape

    for (user, item) in zip(*train.nonzero()):
        # Positive instance
        user_input.append(user)
        item_input.append(item)
        labels.append(1)

        # Negative instances
        for _ in range(num_negatives):
            neg_item = np.random.randint(num_items)
            while (user, neg_item) in zip(*train.nonzero()):
                neg_item = np.random.randint(num_items)
            user_input.append(user)
            item_input.append(neg_item)
            labels.append(0)

    return user_input, item_input, labels


if __name__ == '__main__':
    args = parse_args()
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    topK = 10
    print(f"MLP arguments: {args}")
    model_out_file = f"Pretrain/{args.layers}_MLP_{int(time())}.pth"
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.path)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print(f"Load data done [{time() - t1:.1f} s]. #user={num_users}, #item={num_items}, "
          f"#train={train.nnz}, #test={len(testRatings)}")
    
    # Build model
    model = MLPModel(num_users, num_items, layers).to(device)


    # Set optimizer
    optimizer = {
        "adagrad": optim.Adagrad,
        "rmsprop": optim.RMSprop,
        "adam": optim.Adam,
        "sgd": optim.SGD,
    }.get(learner.lower(), optim.SGD)(model.parameters(), lr=learning_rate)

    # Check Init performance
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print(f'Init: HR = {hr:.4f}, NDCG = {ndcg:.4f} [{time() - t1:.1f}]')

    # Print to verify where the model is located (GPU or CPU)
    print(f"Model is on device: {next(model.parameters()).device}")
    
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        user_input, item_input, labels = get_train_instances(train, num_negatives)

        # Convert to tensors
        user_input = torch.LongTensor(user_input).to(device)
        item_input = torch.LongTensor(item_input).to(device)
        labels = torch.FloatTensor(labels).to(device)  # BCE Loss expects float

        # Training
        model.train()
        optimizer.zero_grad()
        predictions = model(user_input, item_input)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()

        t2 = time()

        # Evaluation
        if epoch % verbose == 0:
            model.eval()
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print(f'Iteration {epoch} [{t2 - t1:.1f} s]: HR = {hr:.4f}, NDCG = {ndcg:.4f}, loss = {loss.item():.4f}')
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                torch.save(model.state_dict(), model_out_file)

    print(f"End. Best Iteration {best_iter}:  HR = {best_hr:.4f}, NDCG = {best_ndcg:.4f}.")
