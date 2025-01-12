import math
import heapq # for retrieval topK
import multiprocessing
import numpy as np
from time import time
#from numba import jit, autojit

# Global variables that are shared across processes
_model = None
_testRatings = None
_testNegatives = None
_K = None

def evaluate_model(model, testRatings, testNegatives, K):
    """
    Evaluate the performance (Normalized Precision@k, NDCG@k) of top-K recommendation
    Return: score of each test rating.
    """
    global _model
    global _testRatings
    global _testNegatives
    global _K
    _model = model
    _testRatings = testRatings
    _testNegatives = testNegatives
    _K = K
        
    hits, ndcgs, precisions, recalls = [],[], [], []
    for idx in range(len(_testRatings)):
        (hr,ndcg,p,r) = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)   
        precisions.append(p)
        recalls.append(r)   
    return (hits, ndcgs,precisions,recalls)

def eval_one_rating(idx):
    rating = _testRatings[idx]
    items = _testNegatives[idx]
    u = rating[0]
    gtItems = rating[1:]
    items += gtItems
    # Get prediction scores
    map_item_score = {}
    users = np.full(len(items), u, dtype = 'int32')
    predictions = _model.predict(users, np.array(items), batch_size=100)
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i].max()
    items.pop()
    
    # Evaluate top rank list
    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHR(ranklist, gtItems)
    ndcg = getNDCG(ranklist, gtItems)
    precision = get_precision(ranklist, gtItems)
    recall = get_recall(ranklist, gtItems)
    return (hr, ndcg, precision, recall)

def getHR(ranklist, gtItems):
    for item in ranklist:
        if item in gtItems:
            return 1
    return 0

def get_precision(ranklist, gtItems):
    relevant = 0
    for item in ranklist:
        if item in gtItems:
            relevant += 1
    return relevant / len(ranklist)

def getNDCG(ranklist, gtItems):
    """
    Calcule la métrique NDCG pour une liste ordonnée (ranklist) et les items pertinents (gtItems).

    Args:
        ranklist: Liste des items prédits, ordonnée par pertinence.
        gtItems: Liste des items pertinents (taille fixe de 2).

    Returns:
        float: La métrique NDCG normalisée.
    """
    dcg = 0.0
    for i, item in enumerate(ranklist):
        if item in gtItems:
            dcg += math.log(2) / math.log(i + 2)  # DCG pour cet item trouvé
    
    # Calcul de l'IDCG (DCG idéal)
    idcg = sum(math.log(2) / math.log(i + 2) for i in range(len(gtItems)))

    return dcg / idcg if idcg > 0 else 0.0

def get_recall(ranklist, gtItems):
    relevant = 0
    for item in ranklist:
        if item in gtItems:
            relevant += 1
    return relevant / len(gtItems)



#### For Matrix factorization binary case : 
def evaluate_metrics(test_data, P, Q, user_id_map, item_id_map, K=10):
    """
    Evaluate metrics (HR@K, NDCG@K, Precision@K, Recall@K) for a leave-2-out test set.
    
    Parameters:
    - test_data: DataFrame containing test samples with the following columns:
        'user', 'id1' (positive item 1), 'id2' (positive item 2), ..., 'negative_1', ..., 'negative_99'
    - P: User latent factor matrix
    - Q: Item latent factor matrix
    - user_id_map: Dictionary mapping user IDs to indices
    - item_id_map: Dictionary mapping item IDs to indices
    - K: Number of top items to consider for metrics
    
    Returns:
    - hr: Average Hit Rate at K
    - ndcg: Average Normalized Discounted Cumulative Gain at K
    - precision: Average Precision at K
    - recall: Average Recall at K
    """
    hr_list = []
    ndcg_list = []
    precision_list = []
    recall_list = []

    for _, row in test_data.iterrows():
        user_id = row['user']
        positive_items = [row['id1'], row['id2']]
        
        # Extract negative items from the row
        negative_items = row[7:].values  # Assuming negatives start from the 8th column

        user_idx = user_id_map[user_id]
        
        # Create a list of candidate items (two positives + negatives)
        candidate_items = positive_items + list(negative_items)
        candidate_indices = [item_id_map[item] for item in candidate_items]
        
        # Compute scores for all candidate items
        scores = [np.dot(P[user_idx], Q[item_idx]) for item_idx in candidate_indices]
        
        # Rank items by their predicted scores
        ranked_indices = np.argsort(scores)[::-1]  # Descending order of scores
        ranked_items = [candidate_items[i] for i in ranked_indices[:K]]  # Top-K items

        # Compute metrics for the current user
        hr = getHR(ranked_items, positive_items)
        ndcg = getNDCG(ranked_items, positive_items)
        precision = get_precision(ranked_items, positive_items)
        recall = get_recall(ranked_items, positive_items)

        hr_list.append(hr)
        ndcg_list.append(ndcg)
        precision_list.append(precision)
        recall_list.append(recall)

    # Compute average metrics across all users
    avg_hr = np.mean(hr_list)
    avg_ndcg = np.mean(ndcg_list)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)

    return avg_hr, avg_ndcg, avg_precision, avg_recall