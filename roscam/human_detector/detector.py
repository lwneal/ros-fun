import sys

import numpy as np
from PIL import Image

sys.path.append('../resnetd')
from vision_client import resnet


def load_model():
    from keras.models import Sequential
    from keras.layers import Convolution2D, ZeroPadding2D
    model = Sequential()
    input_shape = (None, None, 2048)
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(1, 3, 3, activation='sigmoid', name='output'))
    return model


def human_detector(model, visual_activations):
    # Input: batch_size x height x width x 2048
    visual_activations = np.expand_dims(visual_activations, axis=0)
    preds = model.predict(visual_activations)
    # Output: height x width
    return preds.reshape(preds.shape[1:-1])


def save_as_jpg(detection, filename):
    img = Image.fromarray(detection * 255.0)
    width, height = img.size
    img = img.resize((width * 32, height * 32))
    img.convert('RGB').save(filename)


if __name__ == '__main__':
    jpg_data = open(sys.argv[1]).read()
    model = load_model()

    visual_activations = resnet(jpg_data)
    mask = human_detector(model, visual_activations)
    save_as_jpg(mask, sys.argv[2])
