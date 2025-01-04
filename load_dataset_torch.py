import pandas as pd
import scipy.sparse as sp
import numpy as np
import torch

class Dataset:
    """
    Class for loading and managing training and test datasets
    """

    def __init__(self, path, device='gpu'):
        """
        Constructor
        """
        self.device = device
        self.trainMatrix = self.load_rating_file_as_tensor(path + "train.csv")
        self.testRatings = self.load_rating_file_as_list(path + "test.csv")
        self.testNegatives = self.load_negative_file(path + "test_negative.csv")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        """
        Load ratings from a CSV file and return them as a list.
        Each entry is a list containing [user, item].
        """
        df = pd.read_csv(filename)
        rating_list = df[['user', 'id']].values.tolist()
        return rating_list
    
    def load_negative_file(self, filename):
        """
        Load negative samples from a CSV file.
        Each row contains a user and a list of negative items.
        """
        df = pd.read_csv(filename)
        rating_list = []

        # Iterate over each row in the dataframe
        for _, row in df.iterrows():
            # Collect the negative items (all columns after 'id' and 'timestamp' are considered negatives)
            negative_items = row[4:].values.tolist()
            rating_list.append(negative_items)
        
        return rating_list
    
    def load_rating_file_as_tensor(self, filename):
        """
        Read a CSV file containing ratings and return a PyTorch tensor.
        The tensor contains 1 for positive interactions, 0 otherwise.
        """
        df = pd.read_csv(filename)
        num_users = df['user'].max() + 1
        num_items = df['id'].max() + 1

        # Create a dense tensor
        mat = torch.zeros((num_users, num_items), dtype=torch.float32, device=self.device)
        
        for _, row in df.iterrows():
            user, item, rating = int(row['user']), int(row['id']), float(row['rating'])
            mat[user, item] = rating
        
        return mat