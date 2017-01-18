import sys
import numpy as np

GLOVE_FILENAME = 'glove.6B.50d.txt'
START_TOKEN = '--'
END_TOKEN = '...'
UNKNOWN_TOKEN = 'unk'

glove_dict = None
word_list = []


def init():
    global glove_dict
    global word_list
    if glove_dict is None:
        glove_dict, word_list = load_glove_vectors(GLOVE_FILENAME)


def load_glove_vectors(filename, scale_factor=1.0):
    word2vec = {}
    word_list = []
    line_count = sum(1 for line in open(filename))
    sys.stdout.write("\n")
    i = 0
    for line in open(filename).readlines():
        tokens = line.strip().split()
        if len(tokens) < 2:
            print("Warning, error parsing word vector on line: {}".format(line))
            continue
        word = tokens[0]
        word_list.append(word)
        vector = np.asarray([[float(n) for n in tokens[1:]]])[0] * scale_factor
        index = i
        word2vec[word] = (vector, i)
        if i % 1000 == 0:
            sys.stdout.write("\rProcessed {} {}/{} ({:.01f} percent)    ".format(
                filename, i, line_count, 100.0 * i / line_count))
            sys.stdout.flush()
        i += 1
    sys.stdout.write("\n")

    print("Loaded word2vec dictionary for {} words".format(len(word2vec)))
    return word2vec, word_list


def get_dimensionality():
    init()
    unk, idx = glove_dict[UNKNOWN_TOKEN]
    return unk.shape[0]


def vectorize(text):
    init()
    text = preprocess(text)
    word_vectors = [glove_dict[START_TOKEN]]
    for word in text.split():
        word = glove_dict.get(word, glove_dict[UNKNOWN_TOKEN])
        word_vectors.append(word)
    word_vectors.append(glove_dict[END_TOKEN])
    return word_vectors


def preprocess(text):
    return text.lower().replace('?', '').replace(',', ' ')


def pad_to_length(word_vectors, desired_length):
    wordvec_dim = word_vectors.shape[1]
    padding_row_count = desired_length - word_vectors.shape[0]
    if padding_row_count > 0:
        padding = np.zeros((padding_row_count, wordvec_dim))
        word_vectors = np.concatenate((padding, word_vectors))
    return word_vectors[-desired_length:]


def onehot(indices):
    onehot = np.zeros((len(glove_dict), len(indices)))
    for (i, idx) in enumerate(indices):
        onehot[idx, i] = 1.0
    return onehot
