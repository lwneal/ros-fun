"""
A fully convolutional network that takes as input the output of ResNet50,
and outputs a single scalar value for each location.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def build_model(filename=None):
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
