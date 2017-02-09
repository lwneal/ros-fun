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
from networks.mao_net import BATCH_SIZE
from interfaces.image_caption import VOCABULARY_SIZE


def extract_visual_features(jpg_data, box):
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    return image_caption.extract_features_from_preds(preds, width, height, box)


def example_generator(idx):
    # NOTE: Reset the LSTM state after each <end> token is output
    jpg_data, box, text = dataset_grefexp.random_generation_example()
    img_features = extract_visual_features(jpg_data, box)
    words = nlp_api.words_to_onehot(text)
    #print("Generator {}: {}".format(idx, nlp_api.onehot_to_words(words)))
    for word in words:
        yield np.concatenate((img_features, word))
    # Right-pad the output with zeros, which will be masked out
    while True:
        empty = np.zeros(VOCABULARY_SIZE)
        yield np.concatenate((img_features, empty))


def training_batch_generator(**kwargs):
    # Create BATCH_SIZE separate stateful generators
    generators = [example_generator(i) for i in range(BATCH_SIZE)]

    # Y is the target word to predict
    Y = np.array([next(g) for g in generators])
    while True:
        # X is the last word that was output
        X = Y.copy()
        # Input is visual+word, output is word
        Y = np.array([next(g) for g in generators])
        yield np.expand_dims(X, axis=1), Y[:, 4101:]


def manywarm_to_onehot(X, offset=4101):
    # Squash the many-warm softmax output into a one-hot input
    word_idx = np.argmax(X[:,offset:], axis=1)
    X[:,offset:] = 0
    for i in range(BATCH_SIZE):
        X[i, offset + word_idx[i]] = 1.0
    return X


def demonstrate(model, gen):
    DEMO_LEN = 10
    words = np.zeros((DEMO_LEN, BATCH_SIZE, image_caption.VOCABULARY_SIZE))
    X, _ = next(gen)
    visual = X[:,0,:4101]
    #words[0,:,:] = seed_words
    words[0,:,2] = 1.0
    for i in range(1, DEMO_LEN):
        visual_input = np.expand_dims(visual,axis=1)
        word_input = np.expand_dims(words[i-1], axis=1)
        words[i] = model.predict([visual_input, word_input])[:,0,:]
        #print("Predict: {} -> {}".format(np.argmax(words[i-1][0]), np.argmax(words[i][0])))

    print("Demonstration on {} images:".format(BATCH_SIZE))
    for i in range(BATCH_SIZE):
        print nlp_api.onehot_to_words(words[:,i])


def print_weight_stats(model):
    for w in model.get_weights():
        print w.shape, w.min(), w.max()


def train(model, **kwargs):
    start_time = time.time()

    print("Weights min/max:")
    print_weight_stats(model)
    iters = 20
    loss = 0
    for i in range(iters):
        model.reset_states()
        gen = training_batch_generator(**kwargs)
        for i in range(10):
            X, Y = next(gen)
            Y = np.expand_dims(Y, axis=1)
            visual = X[:,:,:4101]
            language = X[:,:,4101:]
            loss += model.train_on_batch([visual, language], Y)
            #print("Train: {} -> {}".format(np.argmax(language[0,0,:]), np.argmax(Y[0,0,:])))
    model.reset_states()
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
