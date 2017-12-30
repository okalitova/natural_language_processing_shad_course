import sys
import numpy as np
from models import PriorModel # <-- Implemented as a uniform distribution.
from models import TranslationModel # <-- Not implemented 
from models import TransitionModel # <-- You will need this for an HMM.
from utils import read_all_tokens, output_alignments_per_test_set
from tqdm import tqdm
import unidecode
from collections import Counter
import re

def get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model):
    "Compute the posterior alignment probability p(a_j=i | f, e) for each target token f_j."
    # alignment_posteriors = np.zeros((len(trg_tokens), len(src_tokens)))
    # log_likelihood = 0.0
    pr_a = prior_model.get_parameters_for_sentence_pair(len(src_tokens), len(trg_tokens))
    pr_f_e = translation_model.get_parameters_for_sentence_pair(src_tokens, trg_tokens)
    mmult = pr_a * pr_f_e
    ssum = np.sum(mmult, axis=0)
    alignment_posteriors = mmult / ssum[None, :]
    log_likelihood = np.sum(np.log(ssum))
    return alignment_posteriors.T, log_likelihood

def collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model):
    "E-step: infer posterior distribution over each sentence pair and collect statistics."
    corpus_log_likelihood = 0.0
    for src_tokens, trg_tokens in tqdm(zip(src_corpus, trg_corpus)):
        # Infer posterior
        alignment_posteriors, log_likelihood = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model)
        # Collect statistics in each model.
        prior_model.collect_statistics(len(src_tokens), len(trg_tokens), alignment_posteriors)
        translation_model.collect_statistics(src_tokens, trg_tokens, alignment_posteriors)
        # Update log prob
        corpus_log_likelihood += log_likelihood
    return corpus_log_likelihood

def estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations):
    "Estimate models iteratively."
    for iteration in range(num_iterations):
        print("iteration: ", iteration)
        # E-step
        corpus_log_likelihood = collect_expected_statistics(src_corpus, trg_corpus, prior_model, translation_model)
        print("E step is done")
        # M-step
        prior_model.recompute_parameters()
        translation_model.recompute_parameters()
        print("M step is done")
        if iteration > 0:
            print("corpus log likelihood: %1.3f" % corpus_log_likelihood)
    return prior_model, translation_model

def align_corpus(src_corpus, trg_corpus, prior_model, translation_model):
    "Align each sentence pair in the corpus in turn."
    alignments = []
    for src_tokens, trg_tokens in zip(src_corpus, trg_corpus):
        posteriors, _ = get_alignment_posteriors(src_tokens, trg_tokens, prior_model, translation_model)
        alignments.append(np.argmax(posteriors, 1))
    return alignments

def initialize_models(src_corpus, trg_corpus):
    prior_model = PriorModel(src_corpus, trg_corpus)
    translation_model = TranslationModel(src_corpus, trg_corpus)
    return prior_model, translation_model

def normalize(src_corpus, trg_corpus, src_tags, trg_tags, trg_lemmas):
    for i in range(len(src_corpus)):
        for j in range(len(src_corpus[i])):
            src_corpus[i][j] = src_corpus[i][j].lower()
            src_corpus[i][j] = unidecode.unidecode(src_corpus[i][j])
            src_corpus[i][j] = src_corpus[i][j][:4]
    for i in range(len(trg_corpus)):
        for j in range(len(trg_corpus[i])):
            trg_corpus[i][j] = trg_lemmas[i][j].lower()
            trg_corpus[i][j] = trg_corpus[i][j][:4]
            trg_corpus[i][j] = unidecode.unidecode(trg_corpus[i][j])
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
    prior_model, translation_model = initialize_models(src_corpus, trg_corpus)
    prior_model, translation_model = estimate_models(src_corpus, trg_corpus, prior_model, translation_model, num_iterations)    
    alignments = align_corpus(src_corpus, trg_corpus, prior_model, translation_model)
    output_alignments_per_test_set(alignments, output_prefix)
