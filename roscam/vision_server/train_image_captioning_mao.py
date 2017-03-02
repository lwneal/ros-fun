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

from keras import models
from keras import layers
# TODO: move to shared?
from nlp_server import word_vector

from shared import util
from shared import nlp_api
from shared.visualizer import Visualizer
import resnet
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net
from networks.mao_net import BATCH_SIZE, WORDVEC_DIM, IMAGE_FEATURE_SIZE, MAX_WORDS
from interfaces.image_caption import VOCABULARY_SIZE


def extract_visual_features(jpg_data, box):
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    return image_caption.extract_features_from_preds(preds, width, height, box)


def get_example():
    x_img = np.zeros((MAX_WORDS, IMAGE_FEATURE_SIZE))
    x_words = np.zeros((MAX_WORDS,), dtype=int)
    # Start token
    x_words[:] = 2
    y = np.zeros(VOCABULARY_SIZE)

    jpg_data, box, text = dataset_grefexp.random_generation_example()

    img_features = extract_visual_features(jpg_data, box)
    x_img[:,:] = img_features

    # Train on one word in the sentence
    _, indices = nlp_api.words_to_vec(text)
    end_idx = random.randint(1, len(indices) - 1)
    start_idx = max(0, end_idx - MAX_WORDS)
    word_count = end_idx - start_idx

    # Input is a sequence of integers
    x_words[-word_count:] = indices[start_idx: end_idx]
    target_word = indices[end_idx]
    # Output is a one-hot vector
    y[target_word] = 1.0
    return x_img, x_words, y


def get_batch(**kwargs):
    X_img = np.zeros((BATCH_SIZE, MAX_WORDS, IMAGE_FEATURE_SIZE))
    X_word = np.zeros((BATCH_SIZE, MAX_WORDS), dtype=int)
    Y = np.zeros((BATCH_SIZE, VOCABULARY_SIZE))
    for i in range(BATCH_SIZE):
        X_img[i], X_word[i], Y[i] = get_example()
    return X_img, X_word, Y


def demonstrate(model, all_zeros=False):
    X_img, X_word, Y = get_batch()

    if all_zeros:
        X_word[:,:] = 2  # START_TOKEN_IDX

    visualizer = Visualizer(model)
    # Given some words, generate some more words
    for i in range(0, MAX_WORDS-1):
        next_word = model.predict([X_img, X_word])
        X_word = np.roll(X_word, -1, axis=1)
        X_word[:,-1] = np.argmax(next_word, axis=1)

    print("Model activations")
    visualizer.run([X_img, X_word])

    print("Demonstration on {} images:".format(BATCH_SIZE))
    for i in range(BATCH_SIZE):
        print nlp_api.indices_to_words(X_word[i])
    

def print_weight_stats(model):
    for w in model.get_weights():
        print w.shape, w.min(), w.max()

def print_words(X, Y):
    last_column = np.expand_dims(np.argmax(Y,axis=1),axis=0)
    indices = np.concatenate((X, last_column.transpose()), axis=1)
    for i in range(len(indices)):
        print(nlp_api.indices_to_words(indices[i]))

def train(model, **kwargs):
    start_time = time.time()
    iters = 64
    loss = 0
    for i in range(iters):
        X_img, X_word, Y = get_batch(**kwargs)
        loss += model.train_on_batch([X_img, X_word], Y)
    #print("Expected:")
    #print_words(X_word, Y)
    #print("Actual:")
    #print_words(X_word, model.predict([X_img, X_word]))
    loss /= iters
    print("Finished training for {} batches. Avg. loss: {}".format(iters, loss))

    print("Demonstration from mid-sentence")
    demonstrate(model)
    print("Demonstration from start token")
    demonstrate(model, all_zeros=True)

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
    model.compile(loss='mse', optimizer='rmsprop', learning_rate=learning_rate, decay=.001)
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
