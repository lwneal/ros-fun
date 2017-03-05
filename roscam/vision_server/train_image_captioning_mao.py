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

from shared import util
from shared import nlp_api
from shared.visualizer import Visualizer
import resnet
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net
from networks.mao_net import WORDVEC_DIM, IMAGE_FEATURE_SIZE
from interfaces.image_caption import VOCABULARY_SIZE
from interfaces.image_caption import predict
from interfaces import image_caption
from shared.nlp_api import START_TOKEN_IDX


def get_example():
    x_img, indices = image_caption.example_mao()

    # Important: Select a subset slice of input text
    start_idx = 0
    #start_idx = np.random.randint(0, len(indices) - 2)
    word_count = np.random.randint(1, len(indices) - start_idx)
    x_words = np.array(indices[start_idx:start_idx + word_count])

    # Output is a one-hot vector
    target_word = indices[start_idx+word_count]
    y = np.zeros(VOCABULARY_SIZE)
    y[target_word] = 1.0

    x_img = np.expand_dims(x_img, axis=0)
    x_img = np.repeat(x_img, repeats=word_count, axis=0)
    return x_img, x_words, y


def get_batch(**kwargs):
    while True:
        x_img, x_words, y = get_example()
        X_img = np.expand_dims(x_img, axis=0)
        X_words = np.expand_dims(x_words, axis=0)
        Y = np.expand_dims(y, axis=0)
        yield [X_img, X_words], Y


def demonstrate(model, visualize=False):
    if visualize:
        visualizer = Visualizer(model)
    x_img, x_word, y = get_example()
    output = predict(model, x_img[:1], [START_TOKEN_IDX])
    if visualize:
        visualizer.run([np.expand_dims(x_img[:1], axis=0), np.expand_dims([START_TOKEN_IDX], axis=0)])
    print(nlp_api.indices_to_words(output))


# TODO: Separate all code below this line into generic model trainer
def train(model, **kwargs):
    start_time = time.time()
    next(get_batch(**kwargs))
    hist = model.fit_generator(get_batch(**kwargs), samples_per_epoch=2**8, nb_epoch=1)

    for _ in range(2):
        demonstrate(model, visualize=True)

    info = {
        'start_time': start_time,
        'duration': time.time() - start_time,
        'loss': hist.history['loss'],
        'training_kwargs': kwargs,
    }
    print(info)
    return info


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
    model.compile(loss='mse', optimizer='rmsprop', learning_rate=learning_rate, decay=.0001, metrics=['accuracy'])
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
