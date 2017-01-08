import sys
import math

import numpy as np
from PIL import Image

sys.path.append('../resnetd')
from vision_client import resnet
import dataset_coco


def load_model():
    from keras.models import Sequential
    from keras.layers import Convolution2D, ZeroPadding2D
    model = Sequential()
    input_shape = (None, None, 2048)
    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Convolution2D(1, 3, 3, activation='sigmoid', name='output'))
    model.compile(optimizer='sgd', loss='mse', learning_rate=1.0)
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


def resize_mask(mask, factor=32.0):
    height, width = mask.shape
    new_height = int(math.ceil(height / factor))
    new_width = int(math.ceil(width / factor))
    return imresize(mask, (new_height, new_width))


def rescale(image, shape):
    # TODO: Get rid of imresize
    from scipy.misc import imresize
    return imresize(image, shape).astype(float) / 255.0


def train(model):
    for _ in range(100):
        meta, pixels = dataset_coco.random_image()
        jpg_data = open(meta['filename']).read()
        mask = dataset_coco.human_detection_mask(meta)

        x = resnet(jpg_data)
        print("Resnet input shape {} {} output shape {} {}".format(
            meta['height'], meta['width'], x.shape[1], x.shape[2]))
        height, width, channels = x.shape
        y = rescale(mask, x.shape)

        x = np.expand_dims(x, axis=0)
        y = np.expand_dims(y, axis=2)
        y = np.expand_dims(y, axis=0)
        model.fit(x, y, batch_size=1)

if __name__ == '__main__':
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    jpg_data = open(input_filename).read()
    model = load_model()
    train(model)

    print("Running on test image")
    visual_activations = resnet(jpg_data)
    mask = human_detector(model, visual_activations)
    save_as_jpg(mask, output_filename)
    print("Saved output to {}".format(output_filename))
