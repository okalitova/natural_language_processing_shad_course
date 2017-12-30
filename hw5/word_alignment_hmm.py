import sys
import numpy as np
# from models_hmm import PriorModel # <-- Implemented as a uniform distribution.
from models_hmm import TranslationModel # <-- Not implemented
from models_hmm import TransitionModel # <-- You will need this for an HMM.
from utils import read_all_tokens, output_alignments_per_test_set
from tqdm import tqdm
import unidecode
from collections import Counter
from hmm import forward, backward
import re

def get_alignment_posteriors(src_tokens, trg_tokens, transition_model, translation_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    alignment_posteriors = np.zeros((len(src_tokens), len(src_tokens)))
    log_likelihood = 0.0
    A = transition_model.get_parameters_for_sentence_pair(len(src_tokens))
    O = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)
    pi = np.ones(len(src_tokens)) / len(src_tokens)
    alpha, a = forward((pi, A, O), np.arange(0, len(trg_tokens)))
    beta, b = backward((pi, A, O), np.arange(0, len(trg_tokens)))
    # print(len(src_tokens), len(trg_tokens))
    # print(alpha.shape, A.shape, O.shape, beta.shape)
    for i in range(len(src_tokens)):
        for i_1 in range(len(src_tokens)):
            for j in range(1, len(trg_tokens)):
                alignment_posteriors[i_1][i] += alpha[j-1][i_1] * A[i_1][i] * O[i][j] * beta[j][i]
    return alignment_posteriors / alignment_posteriors.sum(), log_likelihood

def get_posteriors_probs(src_tokens, trg_tokens, transition_model, translation_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    alignment_posteriors = np.zeros((len(src_tokens), len(src_tokens)))
    log_likelihood = 0.0
    A = transition_model.get_parameters_for_sentence_pair(len(src_tokens))
    O = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)
    pi = np.ones(len(src_tokens)) / len(src_tokens)
    alpha, a = forward((pi, A, O), np.arange(0, len(trg_tokens)))
    beta, b = backward((pi, A, O), np.arange(0, len(trg_tokens)))
    # print(len(src_tokens), len(trg_tokens))
    # print(alpha.shape, A.shape, O.shape, beta.shape)
    return alpha * beta / (alpha * beta).sum()

def collect_expected_statistics(src_corpus, trg_corpus, transition_model, translation_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in tqdm(zip(src_corpus, trg_corpus)):
        # Infer posterior
        alignment_posteriors, log_likelihood = get_alignment_posteriors(src_tokens, trg_tokens, transition_model, translation_model)
        # Collect statistics in each model.
        transition_model.collect_statistics(len(src_tokens), alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens, alignment_posteriors)
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood


def estimate_models(src_corpus, trg_corpus, transition_model, translation_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        print("iteration: ", iteration)
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, transition_model, translation_model)
        print("E step is done")
        # M-step
        transition_model.recompute_parameters()
        translation_model.recompute_parameters()
        print("M step is done")
        if iteration > 0:
            print("corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return transition_model, translation_model

def align_corpus(src_corpus, trg_corpus, transition_model, translation_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        # posteriors, _ = get_alignment_posteriors(src_tokens, trg_tokens, transition_model, translation_model)
        posteriors_probs = get_posteriors_probs(src_tokens, trg_tokens, transition_model, translation_model)
        alignments.append(np.argmax(posteriors_probs, 1))
    return alignments

def initialize_models(src_corpus, trg_corpus):
    transition_model = TransitionModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    return transition_model, translation_model

def normalize(src_corpus, trg_corpus, src_tags, trg_tags, trg_lemmas):
    for i in range(len(src_corpus)):
        for j in range(len(src_corpus[i])):
            src_corpus[i][j] = src_corpus[i][j].lower()
            src_corpus[i][j] = unidecode.unidecode(src_corpus[i][j])
            src_corpus[i][j] = src_corpus[i][j][:6]
    for i in range(len(trg_corpus)):
        for j in range(len(trg_corpus[i])):
            trg_corpus[i][j] = trg_lemmas[i][j].lower()
            trg_corpus[i][j] = unidecode.unidecode(trg_corpus[i][j])
            trg_corpus[i][j] = trg_corpus[i][j][:6]
    return src_corpus, trg_corpus

if __name__ == "__main__":
    if not len(sys.argv) == 5:
        print("Usage ./word_alignment.py src_corpus trg_corpus iterations output_prefix.")
        sys.exit(0)
    src_corpus, trg_corpus = read_all_tokens(sys.argv[1]), read_all_tokens(sys.argv[2])
    trg_lemmas = read_all_tokens(sys.argv[2].replace('tokens', 'lemmas'))
    src_tags, trg_tags = read_all_tokens(sys.argv[1].replace('tokens', 'tags')), read_all_tokens(sys.argv[2].replace('tokens', 'tags'))
    src_corpus, trg_corpus = normalize(src_corpus, trg_corpus, src_tags, trg_tags, trg_lemmas)
    num_iterations = int(sys.argv[3])
    output_prefix = sys.argv[4]
    assert len(src_corpus) == len(trg_corpus), "Corpora should be same size!"
    transition_model, translation_model = initialize_models(src_corpus, trg_corpus)
    transition_model, translation_model = estimate_models(src_corpus, trg_corpus, transition_model, translation_model, num_iterations)
    alignments = align_corpus(src_corpus, trg_corpus, transition_model, translation_model)
    output_alignments_per_test_set(alignments, output_prefix)
