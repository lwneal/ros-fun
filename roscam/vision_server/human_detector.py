import time
import sys
import os

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = None


def init(filename=None):
    start_time = time.time()
    global model
    import resnet
    resnet.init()
    model = load_model(filename)
    print("Human detector initialized in {:.3f} sec".format(time.time() - start_time))


def load_model(filename=None):
    if filename:
        from keras.models import load_model
        return load_model(filename)

    from keras.models import Sequential
    from keras.layers import Convolution2D, ZeroPadding2D
    model = Sequential()
    input_shape = (None, None, 2048)
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(64, 3, 3, activation='sigmoid', name='conv1'))
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(1, 3, 3, activation='sigmoid', name='output'))
    model.compile(optimizer='sgd', loss='mse', learning_rate=1.0)
    return model


def run(pixels):
    import resnet
    resnet_preds = resnet.run(pixels)
    inputs = np.expand_dims(resnet_preds, axis=0)
    preds = model.predict(inputs)
    return preds
