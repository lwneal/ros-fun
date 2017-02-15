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
from shared.visualizer import Visualizer
import resnet
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net
from networks.mao_net import BATCH_SIZE, WORDVEC_DIM
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

    # HACK for testing
    idx = random.randint(0,4)
    img_features[:] = 0
    img_features[idx] = 1.
    text = [
        'Hey Jude, take a sad song and make it better.',
        'Wearing a face that she keeps in a jar by the door',
        'Living is easy with eyes closed, misunderstanding all you see',
        "There's nowhere you can be that isn't where you're meant to be",
        'And in the end, the love you take is equal to the love you make',
    ][idx]

    # TODO: faster
    onehots = nlp_api.words_to_onehot(text)
    words, indices = nlp_api.words_to_vec(text)
    #print("Generator {}: {}".format(idx, nlp_api.onehot_to_words(words)))

    for word, onehot in zip(words, onehots):
        model_input = np.concatenate((img_features, word))
        yield model_input, onehot

    # Right-pad the output with zeros, which will be masked out
    while True:
        empty_onehot = np.zeros(VOCABULARY_SIZE)
        img_features[:] = 0  # TODO: do we need to zero both visual and language input for masking to work?
        model_input = np.concatenate((img_features, np.zeros(WORDVEC_DIM)))
        yield model_input, empty_onehot


def training_batch_generator(**kwargs):
    # Create BATCH_SIZE separate stateful generators
    generators = [example_generator(i) for i in range(BATCH_SIZE)]

    X1 = np.zeros((BATCH_SIZE, 1, 4101 + WORDVEC_DIM))
    X2 = np.zeros((BATCH_SIZE, 1, 4101 + WORDVEC_DIM))
    Y1 = np.zeros((BATCH_SIZE, 1, VOCABULARY_SIZE))
    Y2 = np.zeros((BATCH_SIZE, 1, VOCABULARY_SIZE))
    for i in range(len(generators)):
        X2[i, 0], Y2[i, 0] = next(generators[i])

    while True:
        X1 = X2.copy()
        Y1 = Y2.copy()
        for i in range(len(generators)):
            X2[i, 0], Y2[i, 0] = next(generators[i])
        yield X1, Y2


def demonstrate(model):
    word_vectors = np.zeros((WORDS, BATCH_SIZE, WORDVEC_DIM))
    word_idxs = np.zeros((WORDS, BATCH_SIZE), dtype=int)
    gen = training_batch_generator()
    X, _ = next(gen)
    visual = X[:,0,:4101]
    visual_input = np.expand_dims(visual,axis=1)

    # Set first word to the start token
    word_vectors[0] = nlp_api.words_to_vec(['000'])[0][0]
    word_idxs[0, :] = 2


    for i in range(1, WORDS):
        word_input = np.expand_dims(word_vectors[i-1], axis=1)
        model_output = model.predict([visual_input, word_input])[:,0,:]

        from keras import backend as K
        lstm_state = K.get_session().run(model.layers[1].states[0].value())

        # Get the output words indices and back-embed to word vectors
        word_idxs[i] = np.argmax(model_output, axis=1)
        import pdb; pdb.set_trace()
        word_vectors[i] = nlp_api.indices_to_vec(word_idxs[i])

    print("Demonstration on {} images:".format(BATCH_SIZE))
    for i in range(BATCH_SIZE):
        print nlp_api.indices_to_words(list(word_idxs[:,i]))
    
    print("Visualizing activations")
    visualizer = Visualizer(model)
    visualizer.run([visual_input, word_input])


def print_weight_stats(model):
    for w in model.get_weights():
        print w.shape, w.min(), w.max()


def train(model, **kwargs):
    start_time = time.time()

    print("Weights min/max:")
    print_weight_stats(model)
    iters = 8
    loss = 0
    for i in range(iters):
        model.reset_states()
        gen = training_batch_generator(**kwargs)
        for i in range(WORDS):
            X, Y = next(gen)
            visual, language = X[:,:,:4101], X[:,:,4101:]
            loss += model.train_on_batch([visual, language], Y)
            #print("Train: {} -> {}".format(np.argmax(language[0,0,:]), np.argmax(Y[0,0,:])))
    model.reset_states()
    loss /= iters
    print("Finished training for {} batches. Avg. loss: {}".format(iters, loss))

    demonstrate(model)
    model.reset_states()
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
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=learning_rate)
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
