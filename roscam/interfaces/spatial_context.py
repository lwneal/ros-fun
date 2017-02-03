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

def extract_features(resnet_preds, x, y):
    example_x = np.zeros((TIMESTEPS, 2048, 4))
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

