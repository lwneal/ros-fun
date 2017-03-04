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


def extract_visual_features(jpg_data, box):
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    return image_caption.extract_features_from_preds(preds, width, height, box)


def get_example():
    jpg_data, box, text = dataset_grefexp.random_generation_example()
    img_features = extract_visual_features(jpg_data, box)

    # Train on one word in the sentence
    _, indices = nlp_api.words_to_vec(text)
    if len(indices) < 3:
        print("Warning: invalid caption {}".format(text))
        indices = indices + indices
    word_count = np.random.randint(1, len(indices) - 1)

    # Input is a sequence of integers
    x_words = np.array(indices[:word_count])

    # Output is a one-hot vector
    target_word = indices[word_count]
    y = np.zeros(VOCABULARY_SIZE)
    y[target_word] = 1.0

    x_img = np.zeros((len(x_words), IMAGE_FEATURE_SIZE))
    x_img[:,:] = img_features
    return x_img, x_words, y


def get_batch(**kwargs):
    x_img, x_words, y = get_example()
    X_img = np.expand_dims(x_img, axis=0)
    X_words = np.expand_dims(x_words, axis=0)
    Y = np.expand_dims(y, axis=0)
    return X_img, X_words, Y


def demonstrate(model):
    print("Demonstration on {} images:".format(BATCH_SIZE))
    #visualizer = Visualizer(model)
    for _ in range(4):
        x_img, x_word, y = get_example()
        output = mao_net.predict(model, x_img, x_word)
        print(nlp_api.indices_to_words(output))
        #visualizer.run([X_img, X_word])
    

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
    loss /= iters
    print("Finished training for {} batches. Avg. loss: {}".format(iters, loss))

    print("Demonstration from mid-sentence")
    demonstrate(model)

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
    model.compile(loss='mse', optimizer='rmsprop', learning_rate=learning_rate, decay=.0001)
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
