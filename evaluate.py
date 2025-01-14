import math
import heapq # for retrieval topK
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
        hr, ndcg, p, r = eval_one_rating(idx)
        hits.append(hr)
        ndcgs.append(ndcg)   
        precisions.append(p)
        recalls.append(r)   
    return hits, ndcgs, precisions, recalls

def eval_one_rating(idx):
    hr, ndcg, precision, recall = [], [], [], []
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

    # Evaluate top rank list for topK in (1,10)
    for k in  range(1, _K + 1):
        ranklist = heapq.nlargest(k, map_item_score, key=map_item_score.get)
        hr.append(getHR(ranklist, gtItems))
        ndcg.append(getNDCG(ranklist, gtItems))
        precision.append(get_precision(ranklist, gtItems))
        recall.append(get_recall(ranklist, gtItems))
    return hr, ndcg, precision, recall

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