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
from networks.mao_net import BATCH_SIZE, WORDVEC_DIM, IMAGE_FEATURE_SIZE
from interfaces.image_caption import VOCABULARY_SIZE

# Train and predict on sequences of words up to this length
WORDS = 12

def extract_visual_features(jpg_data, box):
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    return image_caption.extract_features_from_preds(preds, width, height, box)


def example_generator(idx):
    jpg_data, box, text = dataset_grefexp.random_generation_example()
    img_features = extract_visual_features(jpg_data, box)

    img_features /= img_features.max()
    text = text + ' 001'
    onehots = nlp_api.words_to_onehot(text)

    # Repeat caption for up to WORDS timesteps
    while True:
        for onehot, nexthot in zip(onehots, onehots[1:]):
            yield img_features, onehot, nexthot


def training_batch_generator(**kwargs):
    generators = [example_generator(i) for i in range(BATCH_SIZE)]

    X_img = np.zeros((BATCH_SIZE, 1, IMAGE_FEATURE_SIZE))
    X_word = np.zeros((BATCH_SIZE, 1, VOCABULARY_SIZE))
    Y = np.zeros((BATCH_SIZE, 1, VOCABULARY_SIZE))
    while True:
        for i in range(len(generators)):
            X_img[i], X_word[i], Y[i] = next(generators[i])
        yield X_img, X_word, Y


def demonstrate(model):
    gen = training_batch_generator()
    x_img, x_word, y = next(gen)
    words = np.zeros((WORDS, BATCH_SIZE, 1, VOCABULARY_SIZE))

    words[0, :] = x_word

    visualizer = Visualizer(model)
    for i in range(1, WORDS):
        words[i,:,0,:] = model.predict([x_img, words[i-1]])[:,0,:]

    print("Model activations")
    visualizer.print_states()
    visualizer.run([x_img, x_word])

    print("Demonstration on {} images:".format(BATCH_SIZE))
    for i in range(BATCH_SIZE):
        indices = np.argmax(words[:, i, 0], axis=-1)
        print nlp_api.indices_to_words(list(indices))
    

def print_weight_stats(model):
    for w in model.get_weights():
        print w.shape, w.min(), w.max()


glove_model = None
def load_glove_model():
    global glove_model
    word_vector.init()
    glove_weights = np.zeros((WORDVEC_DIM, VOCABULARY_SIZE))
    for i in range(VOCABULARY_SIZE):
        glove_weights[:, i] = word_vector.glove_dict[word_vector.idx_word[i]]
    glove_model = models.Sequential()
    bias = np.zeros(VOCABULARY_SIZE)
    glove_model.add(layers.Dense(input_dim=WORDVEC_DIM, output_dim=VOCABULARY_SIZE, weights=[glove_weights, bias]))


def vectors_to_indices(vectors):
    if glove_model is None:
        load_glove_model()
    wordcount, dim = vectors.shape
    assert dim == WORDVEC_DIM
    preds = glove_model.predict(vectors)
    indices = np.argmax(preds, axis=1)
    return indices


def train(model, **kwargs):
    start_time = time.time()
    iters = 16
    loss = 0
    for i in range(iters):
        model.reset_states()
        gen = training_batch_generator(**kwargs)
        for i in range(WORDS):
            x_img, x_word, y = next(gen)
            loss += model.train_on_batch([x_img, x_word], y)
    model.reset_states()
    loss /= iters
    print("Finished training for {} batches. Avg. loss: {}".format(iters, loss))

    demonstrate(model)
    model.reset_states()

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
    model.compile(loss='mse', optimizer='rmsprop', learning_rate=learning_rate)
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
