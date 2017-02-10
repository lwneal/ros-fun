import json
import requests
import pickle
import numpy as np

VOCABULARY_SIZE = 28519  # TODO: get this programaticly
START_TOKEN = '000'
END_TOKEN = '001'
UNKNOWN_TOKEN = 'stuff'
END_TOKEN_IDX = 3

HOSTNAME = 'localhost'

# Input: text, a string
def words_to_vec(text):
    URL = 'http://{}:8010/words_to_vec'.format(HOSTNAME)
    r = requests.get(URL, data = {'text': text})
    words = pickle.loads(r.text)
    vectors, indices = zip(*words)
    return vectors, indices


def indices_to_onehot(indices):
    onehot = np.zeros((len(indices), VOCABULARY_SIZE))
    for (i, idx) in enumerate(indices):
        onehot[i][idx] = 1.0
    return onehot


def onehot_to_indices(onehot):
    indices = []
    word_count, dimensionality = onehot.shape
    for i in range(word_count):
        indices.append(np.argmax(onehot[i]))
    return indices


def indices_to_words(indices):
    URL = 'http://{}:8010/indices_to_words'.format(HOSTNAME)
    indices_text = json.dumps(indices)
    r = requests.get(URL, data = {'indices': indices_text})
    words = r.text
    return words


def onehot_to_words(onehot):
    return indices_to_words(onehot_to_indices(onehot))


def words_to_onehot(words, pad_to_length=None):
    vectors, indices = words_to_vec(words)
    if pad_to_length:
        v = indices_to_onehot(indices)
        word_count = min(v.shape[0], pad_to_length)
        # Fill from left with up to pad_to_length vectors
        ret = np.zeros((pad_to_length, VOCABULARY_SIZE))
        ret[:word_count] = v[:word_count]
        # Fill right with END_TOKEN
        ret[word_count:, END_TOKEN_IDX] = 1.0
        return ret
    else:
        return indices_to_onehot(indices)

# TODO: Should be a single network request
def indices_to_vec(indices):
    return words_to_vec(indices_to_words(indices))
