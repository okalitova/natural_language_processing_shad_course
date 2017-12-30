import math
import numpy as np

def evaluate_dcg(object_vecs, duplicate_idxs, negative_idxs, k_values):
    """ 
    Ranks candidates by their embeddings and evaluates the ranking by DCG metric.
    
    Args:
        object_vecs (ndarray): Embeddings for all objects (questions).
        duplicate_idxs (list([ind1, ind2])): Duplicate indices (as defined by order in object_vecs).
        negative_idxs (list([ind_neg1, ... ind_negN])): Indices of negative objects for each duplicate pair.
        k_values (list): Several sizes of ranked lists for computing DCG@k.
    
    Returns:
    
        dcg_values (list): Computed values of DCG_at_k for each k (averaged over examples).
    """
    
    assert len(duplicate_idxs) == len(negative_idxs)
    
    # List (by a number of queries) of lists (by a number of different k) of dcg_at_k values. 
    dcg_values = []
    index = -1
    eps = 1e-9
    
    for (duplicate_ind1, duplicate_ind2), neg_indxs in zip(duplicate_idxs, negative_idxs):
        negative_size = len(neg_indxs)
        repeated_query = np.repeat(duplicate_ind1, negative_size + 1)
        candidates = np.hstack([duplicate_ind2, neg_indxs])
        
        similarities = []
        for query_indx, candidate_indx in zip(repeated_query, candidates):
            query = np.array(object_vecs[query_indx])
            candidate = np.array(object_vecs[candidate_indx])
            similarity = (query * candidate).sum() / (math.sqrt((query * query).sum()) + eps) / (math.sqrt((candidate * candidate).sum()) + eps)
            similarities.append(similarity)
        similarities = np.array(similarities)    
        args_sim = np.argsort(similarities)[::-1]
        rank_dub = list(args_sim).index(0) + 1
        dcg_values.append([])
        index = index + 1
        for k in k_values:
            if rank_dub <= k:
                dcg_values[index].append(1. / math.log(1 + rank_dub, 2))
            else:
                dcg_values[index].append(0.)
        
    # Average over different queries.
    dcg_values = np.mean(dcg_values, axis=0)
        
    return dcg_values

def question2vec_advanced(questions, embeddings):
    """ 
    Computes question embeddings by averaging word embeddings.
    
    Args:
      questions (list of strings): List of questions to be embedded.
      embeddings (gensim object): Pre-trained word embeddings.
      
    Returns:
      ndarray of shape [num_questions, embed_size] with question embeddings.
    """
    
    questions_embeddings = []
    for question in questions:
        words = question.split(' ')
        average_emb = np.zeros(embeddings.vector_size)
        words_found = 0
        for word in words:
            current_word = word
            while not current_word in embeddings:
                if len(current_word) == 0:
                    break
                current_word = current_word[:-1]
            if len(current_word) != 0:
                average_emb = average_emb + np.array(embeddings[current_word])
                words_found = words_found + 1
        if words_found != 0:
            average_emb = average_emb / words_found
            questions_embeddings.append(average_emb)
        else:
            questions_embeddings.append(np.ones(embeddings.vector_size) / len(words))
    return np.array(questions_embeddings)

