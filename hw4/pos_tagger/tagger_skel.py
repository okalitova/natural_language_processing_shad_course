#!/usr/bin/env python3
import argparse
import collections
import os
import pprint
import re
import numpy as np
import cityhash
import pickle
import sys
import scipy.stats as sps


###############################################################################
#                                                                             #
#                                INPUT DATA                                   #
#                                                                             #
###############################################################################


def read_tags(path):
    with open(path, 'r') as f:
        tags = []
        for line in f:
            tags.append(line[:-1])
        return tags

# Word: str
# Sentence: list of str
TaggedWord = collections.namedtuple('TaggedWord', ['text', 'tag'])
# TaggedSentence: list of TaggedWord
# Tags: list of TaggedWord
# TagLattice: list of Tags


def read_tagged_sentences(path):
    with open(path, 'r') as f:
        taggedSentences = []
        line = f.readline()
        while len(line) != 0:
            if line.find("# text") != -1:
                taggedSentence = []
                line = f.readline()
                while line != '\n':
                    values = line.split('\t')
                    word = values[1]
                    tag = values[3]
                    taggedSentence.append(TaggedWord(word, tag))
                    line = f.readline()
                taggedSentences.append(taggedSentence)
            line = f.readline()
        return taggedSentences

def read_tagged_sentences_for_test(path):
    with open(path, 'r') as f:
        taggedSentences = []
        line = f.readline()
        while len(line) != 0:
            taggedSentence = []
            while line != '\n':
                values = line.split('\t')
                word = values[1]
                tag = values[3]
                taggedSentence.append(TaggedWord(word, tag))
                line = f.readline()
            taggedSentences.append(taggedSentence)
            line = f.readline()
        return taggedSentences


def write_tagged_sentence(tagged_sentence, f):
    for i, tagged_word in enumerate(tagged_sentence):
        f.write(str(i + 1) + "\t" + tagged_word.text + "\t_\t" + tagged_word.tag + "\t_\t_\t_\t_\t_\t_" + "\n")
    f.write("\n")

TaggingQuality = collections.namedtuple('TaggingQuality', ['acc'])

def tagging_quality(ref, out):
    """
    Compute tagging quality and reutrn TaggingQuality object.
    """
    if len(ref) != len(out):
        raise Exception("ref and out should have the same length")
    correct = 0
    nwords = 0
    for i in range(len(ref)):
        if len(ref[i]) != len(out[i]):
            raise Exception("ref and out should have the same length", len(ref[i]), len(out[i]))
        sentence_correct = 0
        for j in range(len(ref[i])):
            if ref[i][j].text != out[i][j].text:
                print("words are different: ", ref[i][j], out[i][j])
            if ref[i][j].tag == out[i][j].tag:
                correct = correct + 1
                sentence_correct = sentence_correct + 1
            nwords = nwords + 1
        # if sentence_correct / len(ref[i]) < 0.8:
        #     print("original: ", ref[i])
        #     print("tagged: ", out[i])
        #     print(sentence_correct / len(ref[i]))
    return TaggingQuality(correct / nwords)

###############################################################################
#                                                                             #
#                             VALUE & UPDATE                                  #
#                                                                             #
###############################################################################


class Value:
    """
    Dense object that holds parameters.
    """

    def __init__(self, n):
        self._value = np.ones(n)
        self._n = n

    def dot(self, update):
        score = 0.
        for key, v in update._update.items():
            score = score + self._value[int(key % self._n)] * v
        return score

    def assign(self, other):
        self._value = other._value

    def assign_mul(self, coeff):
        self._value = self._value * coeff

    def assign_madd(self, x, coeff):
        """
        self = self + x * coeff
        x can be either Value or Update.
        """
        # if x is an Update
        for key, v in x._update.items():
            self._value[int(key % self._n)] = self._value[int(key % self._n)] + v * coeff


class Update:
    """
    Sparse object that holds an update of parameters.
    """

    def __init__(self, hashes=None, values=None):
        if hashes is None and values is None:
            self._update = dict()
        else:
            self._update = dict(zip(hashes, values))

    def assign_mul(self, coeff):
        self._update.update((x, y * coeff) for x, y in self._update.items())

    def assign_madd(self, update, coeff):
        for key, v in update._update.items():
            if key in self._update:
                self._update[key] = self._update[key] + v * coeff
            else:
                self._update[key] = v * coeff


###############################################################################
#                                                                             #
#                                  MODEL                                      #
#                                                                             #
###############################################################################


Features = Update


class LinearModel:
    """
    A thing that computes score and gradient for given features.
    """

    def __init__(self, n):
        self._params = Value(n)

    def params(self):
        return self._params

    def score(self, features):
        return self._params.dot(features)

    def gradient(self, features, score):
        return features


###############################################################################
#                                                                             #
#                                    HYPO                                     #
#                                                                             #
###############################################################################


Hypo = collections.namedtuple('Hypo', ['prev', 'pos', 'tagged_word', 'score'])


###############################################################################
#                                                                             #
#                              FEATURE COMPUTER                               #
#                                                                             #
###############################################################################


def h(x):
    """
    Compute CityHash of any object.
    Can be used to construct features.
    """
    return cityhash.CityHash64(repr(x))


TaggerParams = collections.namedtuple('FeatureParams', [
    'src_window',
    'dst_order',
    'max_suffix',
    'beam_size',
    'nparams'
    ])


class FeatureComputer:
    def __init__(self, tagger_params, source_sentence):
        self._tagger_params = tagger_params
        self._source_sentence = source_sentence

    def compute_features(self, hypo):
        """
        Compute features for a given Hypo and return Update.
        """
        f = []
        if hypo is None:
            return Features(f, np.ones(len(f)))

        f.append(h((hypo.tagged_word.tag, hypo.tagged_word.text)))
        for i in range(1, self._tagger_params.max_suffix + 1):
            if len(hypo.tagged_word.text) >= i:
                f.append(h((hypo.tagged_word.tag, "suffix: " + hypo.tagged_word.text[-i:])))
                f.append(h((hypo.tagged_word.tag, "prefix: " + hypo.tagged_word.text[:i])))
        if re.search(r'[0-9]]', hypo.tagged_word.text) is not None:
            f.append(h((hypo.tagged_word.tag, "contains_number")))
        if re.search(r'-', hypo.tagged_word.text) is not None:
            f.append(h((hypo.tagged_word.tag, "contains_hyphen")))
        if hypo.pos != 0 and re.search(r'[A-Z]', hypo.tagged_word.text) is not None:
            f.append(h((hypo.tagged_word.tag, "contains_uppercase")))
        if hypo.prev is not None:
            f.append(h((hypo.tagged_word.tag, "prev tag: " + hypo.prev.tagged_word.tag)))
            f.append(h((hypo.tagged_word.tag, "prev text: " + hypo.prev.tagged_word.text)))
        if hypo.prev is not None and hypo.prev.prev is not None:
            f.append(h((hypo.tagged_word.tag,
                        "prev prev tag: " + hypo.prev.prev.tagged_word.tag,
                        "prev tag: " + hypo.prev.tagged_word.tag)))
            f.append(h((hypo.tagged_word.tag,
                        "prev prev text: " + hypo.prev.prev.tagged_word.text)))
        if hypo.pos < len(self._source_sentence) - 1:
            f.append(h((hypo.tagged_word.tag,
                        "next text: " + self._source_sentence[hypo.pos + 1].text)))
        if hypo.pos < len(self._source_sentence) - 2:
            f.append(h((hypo.tagged_word.tag,
                        "next text: " + self._source_sentence[hypo.pos + 1].text,
                        "next next text: " + self._source_sentence[hypo.pos + 2].text)))
        return Features(f, np.ones(len(f)))

# tagger_params = TaggerParams(src_window=2,
#                             dst_order=2,
#                             max_suffix=4,
#                             beam_size=3,
#                             nparams=10)
# test_sentence = [TaggedWord(text='I', tag='PRON'),
#                  TaggedWord(text='like', tag='VERB'),
#                  TaggedWord(text='black', tag='ADJ'),
#                  TaggedWord(text='cat', tag='NOUN')]
# feature_computer = FeatureComputer(tagger_params, test_sentence)
# prev_hypo = Hypo(prev=None, pos=0, tagged_word=TaggedWord('I', 'PRON'), score=0)
# hypo = Hypo(prev=prev_hypo, pos=1, tagged_word=TaggedWord('like', 'VERB'), score=0)
# print(feature_computer.compute_features(hypo)._update)

###############################################################################
#                                                                             #
#                                BEAM SEARCH                                  #
#                                                                             #
###############################################################################


class BeamSearchTask:
    """
    An abstract beam search task. Can be used with beam_search() generic
    function.
    """

    def __init__(self, tagger_params, sentence, model, tags):
        self._tagger_params = tagger_params
        self._sentence = sentence
        self._feature_computer = FeatureComputer(tagger_params, sentence)
        self._model = model
        self._tags = tags
        self._num_stacks = len(sentence)
        self._beam_size = self._tagger_params.beam_size

    def num_stacks(self):
        """
        Return total number of stacks.
        """
        return self._num_stacks

    def beam_size(self):
        return self._beam_size

    def expand(self, hypo):
        """
        Given Hypo, return a list of its possible expansions.
        'hypo' might be None -- return a list of initial hypos then.
        """
        hypos = []
        if hypo is None:
            for tag in self._tags:
                tagged_word = TaggedWord(tag=tag, text=self._sentence[0].text)
                initial_hypo = Hypo(prev=hypo, pos=0, tagged_word=tagged_word, score=0)
                features = self._feature_computer.compute_features(initial_hypo)
                score = self._model.score(features)
                hypos.append(Hypo(prev=hypo, pos=0, tagged_word=tagged_word, score=score))
        else:
            for tag in self._tags:
                tagged_word = TaggedWord(tag=tag, text=self._sentence[hypo.pos + 1].text)
                initial_hypo = Hypo(prev=hypo, pos=hypo.pos + 1, tagged_word=tagged_word, score=0)
                features = self._feature_computer.compute_features(initial_hypo)
                score = hypo.score + self._model.score(features)
                hypos.append(Hypo(prev=hypo, pos=hypo.pos + 1, tagged_word=tagged_word, score=score))
        return hypos

    def recombo_hash(self, hypo):
        """
        If two hypos have the same recombination hashes, they can be collapsed
        together, leaving only the hypothesis with a better score.
        """
        pass


def beam_search(beam_search_task):
    """
    Return list of stacks.
    Each stack contains several hypos, sorted by score in descending
    order (i.e. better hypos first).
    """
    stacks = [[] for _ in range(beam_search_task.num_stacks())]
    for step in range(beam_search_task.num_stacks()):
        next_hypos = []
        if step == 0:
            next_hypos = next_hypos + beam_search_task.expand(None)
        else:
            for hypo in stacks[step - 1]:
                next_hypos = next_hypos + beam_search_task.expand(hypo)
        next_hypos.sort(key=lambda next_hypo: -next_hypo.score)
        next_hypos = next_hypos[:beam_search_task.beam_size()]
        stacks[step] = next_hypos
    return stacks

# model = LinearModel(10)
# new_value = Value(10)
# update = Update([0, 1, 3], [2, 2, 2])
# model.params().assign_madd(update, 1.)
# tags = read_tags('data/tags')
# beam_search_task = BeamSearchTask(tagger_params, test_sentence, model, tags)
# stacks = beam_search(beam_search_task)
# print(stacks)
###############################################################################
#                                                                             #
#                            OPTIMIZATION TASKS                               #
#                                                                             #
###############################################################################


class OptimizationTask:
    """
    Optimization task that can be used with sgd().
    """

    def params(self):
        """
        Parameters which are optimized in this optimization task.
        Return Value.
        """
        raise NotImplementedError()

    def loss_and_gradient(self, golden_sentence):
        """
        Return (loss, gradient) on a specific example.

        loss: float
        gradient: Update
        """
        raise NotImplementedError()


class UnstructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self):
        print("UnstructuredPerceptronOptimizationTask")
        # TODO
        pass

    def params(self):
        # TODO
        pass

    def loss_and_gradient(self, golden_sentence):
        # TODO
        pass

class StructuredPerceptronOptimizationTask(OptimizationTask):
    def __init__(self, tagger_params, tags):
        self._tagger_params = tagger_params
        self._model = LinearModel(tagger_params.nparams)
        self._tags = tags

    def params(self):
        return self._model.params()

    def loss_and_gradient(self, golden_sentence):
        # Do beam search.
        beam_search_task = BeamSearchTask(
            self._tagger_params,
            golden_sentence,
            self._model,
            self._tags
        )
        stacks = beam_search(beam_search_task)
        # Compute chain of golden hypos (and their scores!).
        feature_computer = FeatureComputer(self._tagger_params, golden_sentence)

        initial_hypo = Hypo(prev=None, pos=0, tagged_word=golden_sentence[0], score=0)
        features = feature_computer.compute_features(initial_hypo)
        score = self._model.score(features)
        golden_hypo = Hypo(prev=None, pos=0, tagged_word=golden_sentence[0], score=score)

        # update_pos = -1
        max_score = 0.
        max_golden = None
        max_rival = None
        for i in range(1, len(golden_sentence)):
            # if golden_hypo.score < stacks[i - 1][-1].score:
            #     update_pos = i - 1
            #     break
            if max_score < stacks[i - 1][0].score - golden_hypo.score:
                max_score = stacks[i - 1][0].score - golden_hypo.score
                max_golden = golden_hypo
                max_rival = stacks[i - 1][0]
            tagged_word = golden_sentence[i]
            initial_hypo = Hypo(prev=golden_hypo, pos=golden_hypo.pos + 1, tagged_word=tagged_word, score=0)
            features = feature_computer.compute_features(initial_hypo)
            score = golden_hypo.score + self._model.score(features)
            golden_hypo = Hypo(prev=golden_hypo, pos=golden_hypo.pos + 1, tagged_word=tagged_word, score=score)
            # update_pos = i

        if max_score < stacks[len(golden_sentence) - 1][0].score - golden_hypo.score:
            max_score = stacks[len(golden_sentence) - 1][0].score - golden_hypo.score
            max_golden = golden_hypo
            max_rival = stacks[len(golden_sentence) - 1][0]

        golden_head = None
        rival_head = None
        if max_golden is not None:
            golden_head = max_golden
            rival_head = max_rival
        else:
            golden_head = golden_hypo
            rival_head = stacks[len(golden_sentence) - 1][0]

        if golden_head.pos != rival_head.pos:
            raise Exception("update positions do not match")

        loss = rival_head.score - golden_head.score

        # Compute gradient.
        grad = Update()
        while golden_head and rival_head:
            rival_features = feature_computer.compute_features(rival_head)
            grad.assign_madd(self._model.gradient(features=rival_features, score=None), 1.)

            golden_features = feature_computer.compute_features(golden_head)
            grad.assign_madd(self._model.gradient(features=golden_features, score=None), -1.)
            golden_head = golden_head.prev
            rival_head = rival_head.prev
        return loss, grad

# opt_task = StructuredPerceptronOptimizationTask(tagger_params, tags)
# opt_task.params().assign_madd(update, 1.)
# print(test_sentence)
# test_sentence[0] = TaggedWord(text='I', tag='AUX')
# test_sentence[1] = TaggedWord(text='like', tag='CCONJ')
# test_sentence[2] = TaggedWord(text='black', tag='DET')
# test_sentence[3] = TaggedWord(text='cat', tag='SYM')
# loss, grad = opt_task.loss_and_gradient(test_sentence)
# opt_task.params().assign_madd(grad, 0.01)
# print(opt_task.params()._value)
###############################################################################
#                                                                             #
#                                    SGD                                      #
#                                                                             #
###############################################################################


SGDParams = collections.namedtuple('SGDParams', [
    'epochs',
    'learning_rate',
    'minibatch_size',
    'average'
    ])


def make_batches(dataset, minibatch_size):
    """
    Make list of batches from a list of examples.
    """
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    n_batches = (len(indices) + minibatch_size - 1) // minibatch_size
    batches = []
    for i in range(n_batches):
        batches.append(np.array(dataset)[indices][i * minibatch_size:(i + 1) * minibatch_size])
    return batches

def sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn):
    """
    Run (averaged) SGD on a generic optimization task. Modify optimization
    task's parameters.

    After each epoch (and also before and after the whole training),
    run after_each_epoch_fn().
    """
    after_each_epoch_fn()
    for i in range(sgd_params.epochs):
        batches = make_batches(dataset, sgd_params.minibatch_size)
        for batch in batches:
            for sample in batch:
                loss, grad = optimization_task.loss_and_gradient(sample)
                optimization_task.params().assign_madd(grad, -sgd_params.learning_rate)
        after_each_epoch_fn()
    after_each_epoch_fn()

###############################################################################
#                                                                             #
#                                    MAIN                                     #
#                                                                             #
###############################################################################


# - Train - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def TRAIN_add_cmdargs(subp):
    p = subp.add_parser('train')

    p.add_argument('--tags',
        help='tags file', type=str, default='data/tags')
    p.add_argument('--dataset',
        help='train dataset', default='data/en-ud-train.conllu')
    p.add_argument('--dataset-dev',
        help='dev dataset', default='data/en-ud-dev.conllu')
    p.add_argument('--model',
        help='NPZ model', type=str, default='model.npz')
    p.add_argument('--sgd-epochs',
        help='SGD number of epochs', type=int, default=15)
    p.add_argument('--sgd-learning-rate',
        help='SGD learning rate', type=float, default=0.01)
    p.add_argument('--sgd-minibatch-size',
        help='SGD minibatch size (in sentences)', type=int, default=32)
    p.add_argument('--sgd-average',
        help='SGD average every N batches', type=int, default=32)
    p.add_argument('--tagger-src-window',
        help='Number of context words in input sentence to use for features',
        type=int, default=2)
    p.add_argument('--tagger-dst-order',
        help='Number of context tags in output tagging to use for features',
        type=int, default=3)
    p.add_argument('--tagger-max-suffix',
        help='Maximal number of prefix/suffix letters to use for features',
        type=int, default=4)
    p.add_argument('--beam-size',
        help='Beam size (0 means unstructured)', type=int, default=1)
    p.add_argument('--nparams',
        help='Parameter vector size', type=int, default=2**22)

    return 'train'

def TRAIN(cmdargs):
    # Beam size.
    optimization_task_cls = StructuredPerceptronOptimizationTask
    # if cmdargs.beam_size == 0:
    #     cmdargs.beam_size = 1
    #     optimization_task_cls = UnstructuredPerceptronOptimizationTask

    # Parse cmdargs.
    tags = read_tags(cmdargs.tags)
    dataset = read_tagged_sentences(cmdargs.dataset)
    dataset_dev = read_tagged_sentences(cmdargs.dataset_dev)
    params = None
    if os.path.exists(cmdargs.model):
        params = pickle.load(open(cmdargs.model, 'rb'))
    sgd_params = SGDParams(
        epochs=cmdargs.sgd_epochs,
        learning_rate=cmdargs.sgd_learning_rate,
        minibatch_size=cmdargs.sgd_minibatch_size,
        average=cmdargs.sgd_average
        )
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=cmdargs.nparams
        )

    # Load optimization task
    optimization_task = optimization_task_cls(tagger_params, tags)
    if params is not None:
        # print('\n\nLoading parameters from %s\n\n' % cmdargs.model)
        optimization_task.params().assign(params)

    # Validation.
    def after_each_epoch_fn():
        model = LinearModel(cmdargs.nparams)
        model.params().assign(optimization_task.params())
        tagged_sentences = tag_sentences(dataset_dev, tagger_params, model, tags)
        q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset_dev))
        # print()
        print(q)
        # print()

        # Save parameters.
        # print('\n\nSaving parameters to %s\n\n' % cmdargs.model)
        pickle.dump(optimization_task.params(), open(cmdargs.model, 'wb'))

    # Run SGD.
    sgd(sgd_params, optimization_task, dataset, after_each_epoch_fn)


# - Test  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def TEST_add_cmdargs(subp):
    p = subp.add_parser('test')

    p.add_argument('--tags',
        help='tags file', type=str, default='data/tags')
    p.add_argument('--dataset',
        help='test dataset', default='data/en-ud-dev.conllu')
    p.add_argument('--model',
        help='NPZ model', type=str, default='model.npz')
    p.add_argument('--tagger-src-window',
        help='Number of context words in input sentence to use for features',
        type=int, default=2)
    p.add_argument('--tagger-dst-order',
        help='Number of context tags in output tagging to use for features',
        type=int, default=3)
    p.add_argument('--tagger-max-suffix',
        help='Maximal number of prefix/suffix letters to use for features',
        type=int, default=4)
    p.add_argument('--beam-size',
        help='Beam size', type=int, default=1)
    return 'test'


def tag_sentences(dataset, tagger_params, model, tags):
    """
    Tag all sentences in dataset. Dataset is a list of TaggedSentence; while
    tagging, ignore existing tags.
    """
    tagged_dataset = [[0 for j in range(len(dataset[i]))] for i in range(len(dataset))]
    for i in range(len(dataset)):
        beam_search_task = BeamSearchTask(tagger_params, dataset[i], model, tags)
        stacks = beam_search(beam_search_task)
        best_head = stacks[-1][0]
        index = len(dataset[i]) - 1
        while best_head is not None:
            tagged_dataset[i][index] = TaggedWord(text=best_head.tagged_word.text,
                                           tag=best_head.tagged_word.tag)
            best_head = best_head.prev
            index = index - 1
    return tagged_dataset

def TEST(cmdargs):
    # Parse cmdargs.
    tags = read_tags(cmdargs.tags)
    dataset = read_tagged_sentences_for_test(cmdargs.dataset)
    params = pickle.load(open(cmdargs.model, 'rb'))
    tagger_params = TaggerParams(
        src_window=cmdargs.tagger_src_window,
        dst_order=cmdargs.tagger_dst_order,
        max_suffix=cmdargs.tagger_max_suffix,
        beam_size=cmdargs.beam_size,
        nparams=0
        )

    # Load model.
    model = LinearModel(params._value.shape[0])
    model.params().assign(params)

    # Tag all sentences.
    tagged_sentences = tag_sentences(dataset, tagger_params, model, tags)

    # Write tagged sentences.
    with open('en-ud-test-notags.conllu', 'w') as f:
        for tagged_sentence in tagged_sentences:
            write_tagged_sentence(tagged_sentence, f)

    # Measure and print quality.
    # q = pprint.pformat(tagging_quality(out=tagged_sentences, ref=dataset))
    # print(q, file=sys.stderr)
    pass


# - Main  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def main():
    # Create parser.
    p = argparse.ArgumentParser('tagger.py')
    subp = p.add_subparsers(dest='cmd')

    # Add subcommands.
    train = TRAIN_add_cmdargs(subp)
    test = TEST_add_cmdargs(subp)

    # Parse.
    cmdargs = p.parse_args()

    # Run.
    if cmdargs.cmd == train:
        TRAIN(cmdargs)
    elif cmdargs.cmd == test:
        TEST(cmdargs)
    else:
        p.error('No command')

if __name__ == '__main__':
    main()
