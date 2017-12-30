from collections import Counter, defaultdict
import numpy as np
import math

def train_hmm(tagged_sents):
    """
    Calucaltes p(tag), p(word|tag), p(tag|tag) from corpus.

    Args:
        tagged_sents: list of list of tagged tokens. 
            Example: 
            [[('dog', 'NOUN'), ('eats', 'VERB'), ...], ...]

    Returns:
        p_t, p_w_t, p_t_t - tuple of 3 elements:
        p_t - dict(float), tag -> proba
        p_w_t - dict(dict(float), tag -> word -> proba
        p_t_t - dict(dict(float), previous_tag -> tag -> proba
    """
    alpha=1e-24
    counter_tag = Counter()
    counter_tag_tag = Counter()
    counter_tag_word = Counter()
    tags = set()
    words = set()
    p_t_t = defaultdict(dict)
    p_w_t = defaultdict(dict)
    p_t = dict()
    
    for tagged_sent in tagged_sents:
        for i, word_tag in enumerate(tagged_sent):
            word, tag = word_tag
            tags.add(tag)
            words.add(word)
            counter_tag_word[(tag, word)] += 1
            counter_tag[tag] += 1
            if i > 0:
                counter_tag_tag[(tagged_sent[i - 1][1], tag)] += 1
    
    for prev_tag_tag in counter_tag_tag.keys():
        prev_tag, tag = prev_tag_tag
        p_t_t[prev_tag][tag] = (counter_tag_tag[(prev_tag, tag)] + alpha)\
        / (counter_tag[prev_tag] + alpha * len(tags))
    
    for tag_word in counter_tag_word.keys():
        tag, word = tag_word
        p_w_t[tag][word] = (counter_tag_word[(tag, word)] + alpha) \
        / (counter_tag[tag] + alpha * len(words))
        
    for tag in counter_tag.keys():
        p_t[tag] = 1 / len(tags)
    
    return p_t, p_w_t, p_t_t

def viterbi_algorithm(test_tokens_list, p_t, p_w_t, p_t_t):
    """
    Args:
        test_tokens_list: list of tokens. 
            Example: 
            ['I', 'go']
        p_t: dict(float), tag->proba
        p_w_t: - dict(dict(float), tag -> word -> proba
        p_t_t: - dict(dict(float), previous_tag -> tag -> proba

    Returns:
        list of hidden tags
    """
    words = 56057
    INF = 10000000
    # number of states
    s = len(p_t)
    # number of tokens
    n = len(test_tokens_list)
    
    # initialization
    viterbi  = dict()
    backpointers = dict()
    
    # initial prob
    for tag in p_t.keys():
        viterbi[(0, tag)] = math.log(p_t[tag]) + math.log(p_w_t[tag].get(test_tokens_list[0], 1/words))
    
    for i in range(1, n):
        for tag in p_t.keys():
            viterbi[(i, tag)] = -INF
            for prev_tag in p_t.keys():
                if viterbi[(i - 1, prev_tag)] + math.log(p_t_t[prev_tag].get(tag, 1/words)) + math.log(p_w_t[tag].get(test_tokens_list[i], 1/words)) >= viterbi[(i, tag)]:
                    viterbi[(i, tag)] = viterbi[(i - 1, prev_tag)] + math.log(p_t_t[prev_tag].get(tag, 1/words)) + math.log(p_w_t[tag].get(test_tokens_list[i], 1/words))
                    backpointers[(i, tag)] = prev_tag
    
    final_tag = ''
    final_prob = -INF
    for tag in p_t.keys():
        if viterbi[(n - 1, tag)] >= final_prob:
            final_prob = viterbi[(n - 1, tag)]
            final_tag = tag
    
    result = [final_tag]
    tag = final_tag
    for i in range(n - 1, 0, -1):
        result.append(backpointers[(i, tag)])
        tag = backpointers[(i, tag)]
        
    return list(reversed(result))