import sys
import numpy as np

GLOVE_FILENAME = '/home/nealla/models/glove.6B.50d.txt'
#GLOVE_FILENAME = '/home/nealla/models/glove.42B.300d.txt'
START_TOKEN = '000'
END_TOKEN = '001'
UNKNOWN_TOKEN = 'stuff'

glove_dict = None
word_idx = {}
idx_word = {}


def init():
    global glove_dict
    global word_idx
    global idx_word
    if glove_dict is None:
        glove_dict = load_glove_vectors(GLOVE_FILENAME)
        word_idx, idx_word = read_vocabulary()


def read_vocabulary(filename='nlp_server/vocabulary.txt'):
    word_idx = {}
    idx_word = {}
    i = 0
    for word in open(filename).read().splitlines():
        word_idx[word] = i
        idx_word[i] = word
        i += 1
    print("Loaded vocabulary of {} words".format(len(word_idx)))
    return word_idx, idx_word


def load_glove_vectors(filename, scale_factor=1.0):
    word2vec = {}
    line_count = sum(1 for line in open(filename))
    sys.stdout.write("\n")
    i = 0
    for line in open(filename).readlines():
        tokens = line.strip().split()
        if len(tokens) < 2:
            print("Warning, error parsing word vector on line: {}".format(line))
            continue
        word = tokens[0]
        vector = np.asarray([[float(n) for n in tokens[1:]]])[0] * scale_factor
        word2vec[word] = vector
        if i % 1000 == 0:
            sys.stdout.write("\rProcessed {} {}/{} ({:.01f} percent)    ".format(
                filename, i, line_count, 100.0 * i / line_count))
            sys.stdout.flush()
        i += 1
    sys.stdout.write("\n")

    print("Loaded word2vec dictionary for {} words".format(len(word2vec)))
    return word2vec


def get_dimensionality():
    init()
    unk = glove_dict[UNKNOWN_TOKEN]
    return unk.shape[0]


def vectorize(text):
    init()
    word_vectors = []
    for word in text_to_words(text):
        word = glove_dict.get(word, glove_dict[UNKNOWN_TOKEN])
        word_vectors.append(word)
    return np.array(word_vectors)


def text_to_idx(text):
    init()
    words = text_to_words(text)
    indices = [word_idx.get(w, word_idx[UNKNOWN_TOKEN]) for w in words]
    return indices


def indices_to_words(indices):
    words = [idx_word[i] for i in indices]
    return words


import re
import string
pattern = re.compile('[\W_]+')
def text_to_words(text):
    text = pattern.sub(' ', text).lower()
    return [START_TOKEN] + text.split() + [END_TOKEN]


def pad_to_length(word_vectors, desired_length):
    wordvec_dim = word_vectors.shape[1]
    padding_row_count = desired_length - word_vectors.shape[0]
    if padding_row_count > 0:
        padding = np.zeros((padding_row_count, wordvec_dim))
        word_vectors = np.concatenate((padding, word_vectors))
    return word_vectors[-desired_length:]


def onehot(indices):
    onehot = np.zeros((len(word_idx), len(indices)))
    for (i, idx) in enumerate(indices):
        onehot[idx, i] = 1.0
    return onehot
