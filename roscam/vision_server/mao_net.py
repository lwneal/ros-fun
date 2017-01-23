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


def extract_features(img, bbox):
    img_height, img_width, channels = img.shape
    resnet_preds = resnet.run(img)
    return extract_features_from_preds(resnet_preds, img_height, img_width, bbox)


def extract_features_from_preds(resnet_preds, img_height, img_width, bbox):
    preds_height, preds_width, preds_depth = resnet_preds.shape
    assert preds_depth == 2048

    # Select a single 2048-dim vector from the center of the bbox
    # TODO: Or average over all vectors in the bbox?
    x0, x1, y0, y1 = bbox
    center_x = ((x0 + x1) / 2.0)  * (float(preds_width) / img_width)
    center_y = ((y0 + y1) / 2.0)  * (float(preds_height) / img_height)
    center_x = np.clip(center_x, 0, preds_width-1)
    center_y = np.clip(center_y, 0, preds_height-1)
    local_preds = resnet_preds[int(center_y), int(center_x)]

    # Also use global context: average over the image
    avg_resnet_preds = resnet_preds.mean(axis=0).mean(axis=0)

    context_vector = np.zeros((5,))
    x0, x1, y0, y1 = bbox
    # Left
    context_vector[0] = float(x0) / img_width
    # Top
    context_vector[1] = float(y0) / img_height
    # Right
    context_vector[2] = float(x1) / img_width
    # Bottom
    context_vector[3] = float(y1) / img_height
    # Size
    context_vector[4] = float((x1 - x0) * (y1 - y0)) / (img_width*img_height)

    # Output: 2048 + 2048 + 5 = 4101
    return np.concatenate((local_preds, avg_resnet_preds, context_vector))


def build_model():
    # Input: The 4101-dim feature from extract_features
    model = Sequential()
    # TODO: Is this exactly the structure used from Mao et al?
    # https://arxiv.org/abs/1511.02283
    model.add(RepeatVector(MAX_OUTPUT_WORDS, input_shape=(4101,)))
    model.add(LSTM(1024, name='lstm_1', return_sequences=True))
    model.add(TimeDistributed(Dense(1024, name='fc_1')))
    model.add(TimeDistributed(Dense(VOCABULARY_SIZE, name='fc_2')))

    # Output: Prediction among all possible words
    model.add(Activation('softmax', name='softmax_1'))

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=.01)
    return model
