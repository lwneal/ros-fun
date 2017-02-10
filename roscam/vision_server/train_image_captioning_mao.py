#!/usr/bin/env python
import subprocess
import json
import time
import random
import socket
import os
import sys
import math

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from shared import nlp_api
import resnet
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net
from networks.mao_net import BATCH_SIZE, WORDVEC_DIM, IMAGE_FEATURE_SIZE, TIMESTEPS
from interfaces.image_caption import VOCABULARY_SIZE

# Train and predict on sequences of words up to this length
WORDS = 16

def extract_visual_features(jpg_data, box):
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    return image_caption.extract_features_from_preds(preds, width, height, box)


def example_generator(idx):
    # NOTE: Reset the LSTM state after each <end> token is output
    jpg_data, box, text = dataset_grefexp.random_generation_example()
    img_features = extract_visual_features(jpg_data, box)

    # TODO: faster
    onehots = nlp_api.words_to_onehot(text)
    words, indices = nlp_api.words_to_vec(text)
    #print("Generator {}: {}".format(idx, nlp_api.onehot_to_words(words)))

    X = np.zeros((TIMESTEPS, IMAGE_FEATURE_SIZE + WORDVEC_DIM))
    Y = np.zeros((TIMESTEPS, VOCABULARY_SIZE))

    for i in range(1, len(words)):
        X[0] = 0
        Y[0] = 0
        X = np.roll(X, -1)
        Y - np.roll(Y, -1)
        X[-1] = np.concatenate((img_features, words[i-1]))
        Y[-1] = onehots[i]
        yield X, Y

    # Empty, should be masked
    while True:
        X = np.roll(X, -1)
        Y - np.roll(Y, -1)
        # If the network outputs "zucchinis zucchinis zucchinis" then masking is broken
        Y[-1][-1] = 1.0
        yield X, Y


def training_batch_generator(**kwargs):
    # Create BATCH_SIZE separate stateful generators
    generators = [example_generator(i) for i in range(BATCH_SIZE)]

    X = np.zeros((BATCH_SIZE, TIMESTEPS, IMAGE_FEATURE_SIZE + WORDVEC_DIM))
    Y = np.zeros((BATCH_SIZE, TIMESTEPS, VOCABULARY_SIZE))

    while True:
        for i in range(len(generators)):
            X[i], Y[i] = next(generators[i])
        yield X, Y[:,-1,:]


def demonstrate(model, batch_gen):
    # Set first word to the start token
    word_idxs = np.zeros((WORDS, BATCH_SIZE), dtype=int)
    word_idxs[0, :] = 2

    for i in range(1, WORDS):
        X, Y = next(batch_gen)
        visual_input, word_input = X[:,:,:4101], X[:,:,4101:]
        model_output = model.predict([visual_input, word_input])
        for j in range(BATCH_SIZE):
            word_idxs[i, j] = np.argmax(model_output[j])

    print("Demonstration on {} images:".format(BATCH_SIZE))
    for i in range(BATCH_SIZE):
        print nlp_api.indices_to_words(list(word_idxs[:,i]))


def print_weight_stats(model):
    for w in model.get_weights():
        print w.shape, w.min(), w.max()


def train(model, **kwargs):
    start_time = time.time()

    print("Weights min/max:")
    print_weight_stats(model)
    iters = 4
    loss = 0
    for i in range(iters):
        gen = training_batch_generator(**kwargs)
        for i in range(WORDS):
            X, Y = next(gen)
            visual, language = X[:,:,:4101], X[:,:,4101:]
            loss += model.train_on_batch([visual, language], Y)
            #print("Train: {} -> {}".format(np.argmax(language[0,0,:]), np.argmax(Y[0,0,:])))
    loss /= iters
    print("Finished training for {} batches. Avg. loss: {}".format(iters, loss))

    demonstrate(model, gen)

    return {
        'start_time': start_time,
        'duration': time.time() - start_time,
        'loss': float(loss),
        'training_kwargs': kwargs,
    }


def save_model_info(info_filename, info, model_filename):
    if os.path.exists(info_filename):
        return
    info['model_filename'] = model_filename
    info['history'] = []
    with open(info_filename, 'w') as fp:
        fp.write(json.dumps(info, indent=2))


def save_training_info(info_filename, info, model_filename):
    info['checksum'] = util.file_checksum(model_filename)
    data = json.load(open(info_filename))
    data['history'].append(info)
    with open(info_filename, 'w') as fp:
        fp.write(json.dumps(data, indent=2))


if __name__ == '__main__':
    model_filename = sys.argv[1]
    model = mao_net.build_model()
    if os.path.exists(model_filename):
        from shared.serialization import load_weights
        load_weights(model, model_filename)

    info_filename = model_filename.replace('.h5', '') + '.json'
    info = {}

    save_model_info(info_filename, info, model_filename)
    resnet.init()
    # TODO: docopt or argparse
    learning_rate = float(sys.argv[2])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=learning_rate)
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
