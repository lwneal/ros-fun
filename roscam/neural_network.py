import os
import numpy as np
import random
import h5py

from resnet50 import ResNet50
import imagenet_utils

model = ResNet50(weights='imagenet', include_top=False)

def pixels_to_input(pixels):
    x = pixels.astype(np.float)
    # Expected input shape: (batch_size x channels x height x width)
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    x = imagenet_utils.preprocess_input(x)
    return x

def init():
    print("Resnet warming up, this will take ~30 seconds...")
    pixels = np.zeros((480, 640, 3))
    x = pixels_to_input(pixels)
    preds = model.predict(x)
    print preds
    print("Resnet is ready!")
    pass

def run(pixels):
    x = pixels_to_input(pixels)
    preds = model.predict(x)
    print preds.shape
    return preds
