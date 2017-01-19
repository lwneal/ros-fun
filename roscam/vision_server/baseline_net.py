# A toy task for learning spatial context for image caption output

# First we build the network

# Input is four variable-length sequences of ResNet features
# An encoder transforms those to a fixed-size representation
# Then a decoder generates a sequence of output words
import time

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import util
import random

from keras.layers.core import Merge
from keras.models import *
from keras.layers import *

import resnet
resnet.init()

MAX_OUTPUT_WORDS = 12
VOCABULARY_SIZE = 28519  # TODO: Get this from the NLP server


def extract_features(img, x, y, preds=None):
    resnet_preds = preds if preds is not None else resnet.run(img)

    height, width, depth = resnet_preds.shape
    assert depth == 2048

    # Select a single 2048-dim vector
    x = np.clip(x, 0, width-1)
    y = np.clip(y, 0, height-1)
    return resnet_preds[y, x]


def build_model():
    # Input: A single 2048-dim ResNet feature vector
    model = Sequential()
    model.add(LSTM(256, input_shape=(None, 2048)))

    # Output: Prediction among all possible words
    model.add(Dense(VOCABULARY_SIZE))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=.5)
    return model
