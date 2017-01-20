import os
import time
import numpy as np
import random
import h5py

model = None

def pixels_to_input(pixels):
    x = pixels.astype(np.float)
    # Expected input shape: (batch_size x height x width x channels)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def preprocess_input(x, dim_ordering='default'):
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68
    # 'RGB'->'BGR'
    x = x[:, :, :, ::-1]
    return x

def init():
    global model
    if model is None:
        start_time = time.time()
        from resnet50 import ResNet50
        model = ResNet50(weights='imagenet', include_top=False)
        print("Initializing ResNet, please wait...")
        pixels = np.zeros((480, 640, 3))
        x = pixels_to_input(pixels)
        preds = model.predict(x)
        print("Resnet initialized in {:.2f} sec".format(time.time() - start_time))

def run(pixels):
    x = pixels_to_input(pixels)
    preds = model.predict(x)
    # (1, 20, 15, 2048) --> (20, 15, 2048)
    return preds.squeeze(0)
