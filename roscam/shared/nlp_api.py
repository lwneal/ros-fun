import json
import requests
import pickle
import numpy as np

WORDVEC_DIM = 400 * 1000
START_TOKEN = '--'
END_TOKEN = '...'
END_TOKEN_IDX = 435
UNKNOWN_TOKEN = 'unk'

def words_to_vec(text):
    URL = 'http://localhost:8010/words_to_vec'
    r = requests.get(URL, data = {'text': text})
    words = pickle.loads(r.text)
    vectors, indices = zip(*words)
    return vectors, indices


def indices_to_onehot(indices):
    onehot = np.zeros((len(indices), WORDVEC_DIM))
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
    URL = 'http://localhost:8010/indices_to_words'
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
        ret = np.zeros((pad_to_length, WORDVEC_DIM))
        ret[:word_count] = v[:word_count]
        # Fill right with END_TOKEN
        ret[word_count:, END_TOKEN_IDX] = 1.0
        return ret
    else:
        return indices_to_onehot(indices)
