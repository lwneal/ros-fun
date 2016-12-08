#!/bin/env python
"""
Usage:
        model.py train --iters <iterations> [--model <model.h5>]
        model.py test [--model <model.h5>]

Options:
        -i, --iters <iterations>        Number of training iterations [default: 10000]
        -m, --model <model.h5>          Filename of already-trained model to start from
"""
import sys
import time
import random

import docopt
import numpy as np
from PIL import Image

import neural_network
import word_vector
from word_vector import vectorize, pad_to_length

from neural_network import OUTPUT_SHAPE, BATCH_OUTPUT_SHAPE, load_weights

MODEL_FILENAME = 'model.h5'
INPUT_WORD_COUNT = 32


class Model(object):
    def __init__(self, load_from_filename=None, batch_size=16):
        self.batch_size = batch_size
        wordvec_dim = word_vector.get_dimensionality()
        self.neural_network = neural_network.build_model(wordvec_dim, INPUT_WORD_COUNT)
        if load_from_filename:
            self.neural_network.load_weights(load_from_filename)

    def evaluate(self, img, question, verbose=False):
        if verbose:
            print('Looking at image size {}x{} and trying to answer question {}'.format(img.width, img.height, question))
        x = [self.vgg_input(img), self.wordvec_rnn(question)]
        answer = self.neural_network.predict(x)
        return answer.reshape(OUTPUT_SHAPE)

    def vgg_input(self, img):
        height = neural_network.IMG_INPUT_HEIGHT
        width = neural_network.IMG_INPUT_WIDTH
        img_input = np.array(img.resize((height,width), Image.ANTIALIAS), np.float).transpose((2,0,1)) * (1.0/256)
        return img_input.reshape((1,3,height,width))

    def wordvec_rnn(self, question):
        word_vectors = pad_to_length(vectorize(question), INPUT_WORD_COUNT)
        return word_vectors.reshape(1, INPUT_WORD_COUNT, word_vector.get_dimensionality())
