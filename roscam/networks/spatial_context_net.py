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

TIMESTEPS = 8
MAX_OUTPUT_WORDS = 8

def left_pad(input_array, to_size=TIMESTEPS):
    input_size, feature_dim = input_array.shape
    y = np.zeros((to_size, feature_dim))
    if input_size > 0:
        y[-input_size:] = input_array[:to_size]
    return y

def clip(x, minval=0, maxval=32):
    return np.clip(x, minval, maxval)

def pad_preds(preds, to_size=32):
    height, width, depth = preds.shape
    height = min(height, to_size)
    width = min(width, to_size)
    padded = np.zeros((to_size, to_size, depth))
    padded[:height, :width] = preds[:height, :width]
    return padded

def extract_features(img, x, y, preds=None):
    example_x = np.zeros((TIMESTEPS, 2048, 4))
    resnet_preds = preds if preds is not None else resnet.run(img)
    # Pad preds to 32x32
    resnet_preds = pad_preds(resnet_preds, to_size=32)
    x = np.clip(x, 0, 31)
    y = np.clip(y, 0, 31)
    # Extract context, padded and in correct order, from left/right/top/bottom
    example_x[:,:,0] = left_pad(resnet_preds[y,                         clip(x - TIMESTEPS) : x,    :])
    example_x[:,:,1] = left_pad(resnet_preds[y,                         x : clip(x + TIMESTEPS),    :][::-1])
    example_x[:,:,2] = left_pad(resnet_preds[clip(y - TIMESTEPS) : y,   x,                          :])
    example_x[:,:,3] = left_pad(resnet_preds[y : clip(y + TIMESTEPS),   x,                          :][::-1])
    return example_x

def spatialContextNet():
    model = Sequential()
    input_shape = (None, 2048)
    gru = GRU(512, input_shape=input_shape, return_sequences=False)
    model.add(gru)
    model.add(Dense(512, activation='sigmoid'))
    return model

def build_model():
    left = spatialContextNet()
    right = spatialContextNet()
    up = spatialContextNet()
    down = spatialContextNet()

    # Input: Four variable-length sequences of 2048-dim ResNet features
    model = Sequential()
    model.add(Merge([left, right, up, down], mode='concat', concat_axis=1))
    model.add(Dense(1024, activation='relu'))
    model.add(RepeatVector(MAX_OUTPUT_WORDS))
    model.add(GRU(256, return_sequences=True))
    model.add(TimeDistributed(Dense(VOCABULARY_SIZE)))
    # Output: Prediction among all words
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=1.0)
    return model

if __name__ == '__main__':
    model = build_model()
    while True:
        train_one_round(model)
        for i in range(10):
            demonstrate(model)
