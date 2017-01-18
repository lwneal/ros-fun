import json
import requests
import pickle
import numpy as np

def words_to_vec(text):
    URL = 'http://localhost:8010/words_to_vec'
    r = requests.get(URL, data = {'text': text})
    print('got response length {}'.format(len(r.text)))
    words = pickle.loads(r.text)
    vectors, indices = zip(*words)
    return vectors, indices


def indices_to_onehot(indices):
    WORDVEC_DIM = 400 * 1000
    onehot = np.zeros((WORDVEC_DIM, len(indices)))
    for (i, idx) in enumerate(indices):
        onehot[idx][i] = 1.0
    return onehot


def onehot_to_indices(onehot):
    indices = []
    dimensionality, word_count = onehot.shape
    print 'Converting {} one-hot words to indices'.format(word_count)
    for i in range(word_count):
        indices.append(np.argmax(onehot[:,i]))
    return indices


def indices_to_words(indices):
    URL = 'http://localhost:8010/indices_to_words'
    indices_text = json.dumps(indices)
    r = requests.get(URL, data = {'indices': indices_text})
    words = r.text
    return words

