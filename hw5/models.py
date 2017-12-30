# Models for word alignment.
#
# This file contains stubs for three models to use to model word alignments.
# Notation: i = src_index, I = src_length, j = trg_index, J = trg_length.
# 
# (i) TranslationModel models p(f|e).
# (ii) PriorModel models p(i|j, I, J).
# (iii) TransitionModel models p(a_{j} = i|a_{j-1} = k).
#
# Each model stores parameters (probabilities) and statistics (counts) as has: 
# (i) A method to access a single probability: get_xxx_prob(...).
# (ii) A method to get all probabilities for a sentence pair as a numpy array:
# get_parameters_for_sentence_pair(...).
# (iii) A method to accumulate 'fractional' counts: collect_statistics(...).
# (iv) A method to recompute parameters: recompute_parameters(...).

import numpy as np
from collections import defaultdict

class TranslationModel:
    "Models conditional distribution over trg words given a src word."

    def __init__(self, src_corpus, trg_corpus):
        self._trg_given_src_probs = defaultdict(lambda : defaultdict(lambda : 1.0))
        self._src_trg_counts = defaultdict(lambda : defaultdict(lambda : 0.0))

    def get_conditional_prob(self, src_token, trg_token):
        "Return the conditional probability of trg_token given src_token."
        return self._trg_given_src_probs[src_token][trg_token]

    def get_parameters_for_sentence_pair(self, src_tokens, trg_tokens):
        "Return numpy array with t[i][j] = p(f_j|e_i)."
        return np.array([[self.get_conditional_prob(src_token, trg_token)
                          for trg_token in trg_tokens] for src_token in src_tokens])

    def collect_statistics(self, src_tokens, trg_tokens, posterior_matrix):
        "Accumulate counts of translations from posterior_matrix[j][i] = p(a_j=i|e, f)"
        # assert False, "Store fractional counts from posterior matrix here."
        # pass
        for i in range(len(src_tokens)):
            for j in range(len(trg_tokens)):
                self._src_trg_counts[src_tokens[i]][trg_tokens[j]] += posterior_matrix[j][i]

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        # assert False, "Normalize to recompute parameters from counts."
        self._trg_given_src_probs = defaultdict(lambda : defaultdict(lambda : 1.0))
        for src, joint_counts in self._src_trg_counts.items():
            src_count = sum([count for count in joint_counts.values()])
            for trg, joint_count in joint_counts.items():
                if src == trg:
                    self._src_trg_counts[src][trg] += 1000000
            for trg, joint_count in joint_counts.items():
                self._trg_given_src_probs[src][trg] = joint_count / src_count
        self._src_trg_counts = defaultdict(lambda: defaultdict(lambda: 0.0))

class PriorDictIJ(dict):

    def __init__(self, i):
        self._i = i

    def __missing__(self, key):
        j, I, J = key
        s = 0.
        for z in range(J):
            s = s + 1 / (abs(self._i - z) + 2)
        self[key] = 1 / (abs(self._i - j) + 2) / s
        return self[key]

class PriorDict(dict):
    def __missing__(self, key):
        self[key] = PriorDictIJ(key)
        return self[key]

class PriorModel:
    "Models the prior probability of an alignment given only the sentence lengths and token indices."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for a prior model."
        self._i_given_j_probs = PriorDict() # Default Uniform prior.
        self._i_given_j_counts = defaultdict(lambda : defaultdict(lambda : 0.0))

    def get_prior_prob(self, src_index, trg_index, src_length, trg_length):
        "Returns a prior probability based on src and trg indices."
        return self._i_given_j_probs[src_index][(trg_index, src_length, trg_length)]
    
    def get_parameters_for_sentence_pair(self, src_length, trg_length):
        "Return a numpy array with all prior p[i][j] = p(i|j, I, J)."
        return np.array([[self.get_prior_prob(i, j, src_length, trg_length)
                          for j in range(trg_length)] for i in range(src_length)])

    def collect_statistics(self, src_length, trg_length, posterior_matrix):
        "Accumulate counts of alignment events from posterior_matrix[j][i] = p(a_j=i|e, f)"
        # for i in range(src_length):
        #     for j in range(trg_length):
        #         self._i_given_j_counts[(j, src_length, trg_length)][i] += posterior_matrix[j][i]
        pass

    def recompute_parameters(self):
        "Reestimate parameters and reset counters."
        # self._i_given_j_probs = PriorDict()  # Default Uniform prior.
        # for j_I_J, j_I_J_counts in self._i_given_j_counts.items():
        #     i_count = sum([count for count in j_I_J_counts.values()])
        #     for i, i_given_j_I_J_counts in j_I_J_counts.items():
        #         self._i_given_j_probs[i][j_I_J] = i_given_j_I_J_counts / i_count
        # self._i_given_j_counts = defaultdict(lambda : defaultdict(lambda : 0.0))
        pass

class TransitionModel:
    "Models the prior probability of an alignment given the previous token's alignment."

    def __init__(self, src_corpus, trg_corpus):
        "Add counters and parameters here for modeling alignment transitions."
        pass

    def get_parameters_for_sentence_pair(self, src_length):
        "Retrieve the parameters for this sentence pair: A[k, i] = p(a_{j} = i|a_{j-1} = k)"
        pass

    def collect_statistics(self, src_length, transition_posteriors):
        "Accumulate statistics from transition_posteriors[k][i]: p(a_{j} = i, a_{j-1} = k|e, f)"
        pass

    def recompute_parameters(self):
        "Reestimate the parameters and reset counters."
        pass
