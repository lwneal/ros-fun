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


def extract_visual_features(jpg_data, box):
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    return image_caption.extract_features_from_preds(preds, width, height, box)


def example_generator():
    # Each generator produces a never-ending stream of input like this:
    # <start> the cat on the mat <end> <start> the dog on the log <end> <start> the...
    # Note that each of the BATCH_SIZE generators is completely separate
    # NOTE: Reset the LSTM state after each <end> token is output
    while True:
        jpg_data, box, text = dataset_grefexp.random_generation_example()
        img_features = extract_visual_features(jpg_data, box)
        words = nlp_api.words_to_onehot(text)
        for word in words:
            yield np.concatenate((img_features, word))


def training_batch_generator(**kwargs):
    # Create BATCH_SIZE separate stateful generators
    generators = [example_generator() for _ in range(BATCH_SIZE)]

    # Given word n, predict word n+1
    Y = np.array([next(g) for g in generators])
    while True:
        X = Y.copy()
        Y = np.array([next(g) for g in generators])
        # Input is visual+word, output is word
        yield np.expand_dims(X, axis=1), Y[:,4101:]

def demonstrate(model, gen):
    DEMO_LEN = 10
    words = np.zeros((DEMO_LEN, BATCH_SIZE, image_caption.VOCABULARY_SIZE))
    X, _ = next(gen)
    visual = X[:,0,:4101]
    seed_words = X[:,0,4101:]
    words[0,:,:] = seed_words
    for i in range(1, DEMO_LEN):
        prev_word = words[i-1]
        model_input = np.concatenate((visual, prev_word), axis=1)
        model_input = np.expand_dims(model_input, axis=1)
        words[i] = model.predict(model_input)

    for i in range(BATCH_SIZE):
        print nlp_api.onehot_to_words(words[:,i])


def train(model, **kwargs):
    start_time = time.time()
    gen = training_batch_generator(**kwargs)

    for i in range(100):
        X, Y = next(gen)
        loss = model.train_on_batch(X, Y)
        word = nlp_api.onehot_to_words(np.array([Y[0]]))
        if word == '001':
            print('\n'),
        else:
            print(word),
    print()
    print("Finished training for 100 batches. Loss: {}".format(loss))

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
    if os.path.exists(model_filename):
        from keras.models import load_model
        model = load_model(model_filename)
    else:
        model = mao_net.build_model()

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
