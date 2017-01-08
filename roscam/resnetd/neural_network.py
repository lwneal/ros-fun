import os
import time
import numpy as np
import random
import h5py

from resnet50 import ResNet50
from keras import backend as K

model = ResNet50(weights='imagenet', include_top=False)

def pixels_to_input(pixels):
    x = pixels.astype(np.float)
    # Expected input shape: (batch_size x height x width x channels)
    x = np.expand_dims(x, axis=0)
    print "shape is {}".format(x.shape)
    x = preprocess_input(x)
    return x

def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x

def init():
    start_time = time.time()
    print("Initializing ResNet, please wait...")
    pixels = np.zeros((480, 640, 3))
    x = pixels_to_input(pixels)
    preds = model.predict(x)
    print('preds: {}'.format(preds))
    print("Resnet initialized in {:.2f} sec".format(time.time() - start_time))
    pass

def run(pixels):
    x = pixels_to_input(pixels)
    preds = model.predict(x)
    print("Got predictions shape {} with min {} max {}".format(preds.shape, preds.min(), preds.max()))
    return preds
