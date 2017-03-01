import sys
import os
import random
from keras import models
from keras import layers
import numpy as np

import word_vector

text = open(sys.argv[1]).read()
word_vector.init('tiny_vocabulary.txt')
word_indices = word_vector.text_to_idx(text)

BATCH_SIZE = 16
TIMESTEPS = 5
VOCAB_LEN = word_vector.vocabulary_len()

#batch_input_shape = (BATCH_SIZE, TIMESTEPS, VOCAB_LEN)
model = models.Sequential()
model.add(layers.Embedding(VOCAB_LEN, 100))
model.add(layers.LSTM(256))
model.add(layers.Dense(VOCAB_LEN, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=.001)

while True:
    X = np.zeros((BATCH_SIZE, TIMESTEPS))
    Y = np.zeros((BATCH_SIZE, VOCAB_LEN))
    for i in range(BATCH_SIZE):
        idx = random.randint(TIMESTEPS, len(word_indices) - 1)
        # Input is a list of TIMESTEPS integers
        X[i] = word_indices[idx - TIMESTEPS: idx]
        # Output is a one-hot vector of floats length VOCAB_LEN
        Y[i, word_indices[idx]] = 1.0
    model.train_on_batch(X, Y)

    if np.random.rand() > .90:
        preds = model.predict(X)
        i = random.randint(0, BATCH_SIZE-1)
        input_words = ' '.join(word_vector.indices_to_words(X[i]))
        output_word = word_vector.indices_to_words(np.argmax(preds, axis=1))[i]
        correct_word = word_vector.indices_to_words(np.argmax(Y, axis=1))[i]
        print('{} {}'.format(input_words, output_word))
