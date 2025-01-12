import pandas as pd
import scipy.sparse as sp
import numpy as np

class Dataset:
    """
    Class for loading and managing training and test datasets
    """

    def __init__(self, path):
        """
        Constructor
        """
        self.trainMatrix = self.load_rating_file_as_matrix(path + "train.csv")
        self.testRatings = self.load_rating_file_as_list(path + "test_negative_transformed.csv")
        self.testNegatives = self.load_negative_file(path + "test_negative_transformed.csv")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        """
        Load ratings from a CSV file and return them as a list.
        Each entry is a list containing [user, item].
        """
        df = pd.read_csv(filename)
        rating_list = df[['user', 'id1', 'id2']].values.tolist()
        return rating_list
    
    def load_negative_file(self, filename):
        """
        Load negative samples from a CSV file.
        Each row contains a user and a list of negative items.
        """
        df = pd.read_csv(filename)
        rating_list = []

        # Iterate over each row in the dataframe
        for index, row in df.iterrows():
            # Collect the negative items (all columns after 'id' and 'timestamp' are considered negatives)
            negative_items = row[7:].values.tolist()
            
            # Append the user and its negative items
            rating_list.append(negative_items)
        
        return rating_list
    
    def load_rating_file_as_matrix(self, filename):
        """
        Read a CSV file containing ratings and return a dok sparse matrix.
        The matrix contains 1 for positive interactions, 0 otherwise.
        """
        df = pd.read_csv(filename)
        num_users = df['user'].max() + 1
        num_items = df['id'].max() + 1

        # Create a sparse matrix
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)
        
        for _, row in df.iterrows():
            user, item, rating = int(row['user']), int(row['id']), float(row['rating'])
            mat[user, item] = rating
        
        return mat
