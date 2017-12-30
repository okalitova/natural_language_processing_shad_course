import re
from collections import Counter

def get_words_counter(corpus):
    corpus = corpus.replace("n't", " not")
    corpus = corpus.replace("'s", " is")
    words_lower = map(lambda word: word.lower(), re.findall('[a-zA-Z]+', corpus))
    word_lower_counts = Counter(words_lower)
    for word in list(word_lower_counts):
        if word_lower_counts[word] < 3:
            del word_lower_counts[word]
    return word_lower_counts
