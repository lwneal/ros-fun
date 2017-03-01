import sys
import os
import random
import numpy as np
from keras import models
from keras import layers

import word_vector
word_vector.init('vocabulary.txt')

BATCH_SIZE = 1024
TIMESTEPS = 5
VOCAB_LEN = word_vector.vocabulary_len()


def main():
    if sys.argv[1] == 'test':
        test()
    else:
        train()


def build_model():
    model = models.Sequential()
    model.add(layers.Embedding(VOCAB_LEN, 100))
    model.add(layers.LSTM(256))
    model.add(layers.Dense(VOCAB_LEN, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=.001)
    return model


def test(seed_text=None):
    model = models.load_model('model.h5')
    TEST_TIMESTEPS = 32
    X = np.zeros((1, TEST_TIMESTEPS), dtype=int)
    if seed_text is not None:
        X[0,-len(seed_text):] = seed_text
    for i in range(TEST_TIMESTEPS):
        preds = model.predict(X)
        word = np.argmax(preds, axis=1)[0]
        X = np.roll(X, -1, axis=1)
        X[0,-1] = word
    print(' '.join(word_vector.indices_to_words(X[0])))


def train():
    text = open(sys.argv[2]).read()
    model = build_model()
    word_indices = word_vector.text_to_idx(text)
    while True:
        for i in range(10):
            X = np.zeros((BATCH_SIZE, TIMESTEPS))
            Y = np.zeros((BATCH_SIZE, VOCAB_LEN))
            for i in range(BATCH_SIZE):
                idx = random.randint(TIMESTEPS, len(word_indices) - 1)
                # Input is a list of TIMESTEPS integers
                X[i] = word_indices[idx - TIMESTEPS: idx]
                # Output is a one-hot vector of floats length VOCAB_LEN
                Y[i, word_indices[idx]] = 1.0
            model.fit(X, Y, nb_epoch=1, batch_size=32)
        model.save('model.h5')
        test(seed_text=X[0])


if __name__ == '__main__':
    main()
