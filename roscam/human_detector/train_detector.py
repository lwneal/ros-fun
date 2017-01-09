import os
import sys
import math

import numpy as np
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../resnetd'))
from vision_client import resnet
import dataset_coco


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


def human_detector(model, visual_activations):
    # Input: batch_size x height x width x 2048
    if len(visual_activations.shape) == 3:
        visual_activations = np.expand_dims(visual_activations, axis=0)
    preds = model.predict(visual_activations)
    # Output: height x width
    print preds.shape
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


def get_example():
    meta, pixels = dataset_coco.random_image()
    jpg_data = open(meta['filename']).read()
    mask = dataset_coco.human_detection_mask(meta)

    x = resnet(jpg_data)
    y = rescale(mask, x.shape)

    # Fill all images into a max-sized 32x32 activation mask
    x_i = np.zeros((32, 32, 2048))
    x_i[:x.shape[0], :x.shape[1]] = x

    y_i = np.zeros((32, 32, 1))
    y_i[:y.shape[0], :y.shape[1], 0] = y
    return x_i, y_i


def get_batch(batch_size=32):
    X = []
    Y = []
    for _ in range(batch_size):
        x_i, y_i = get_example()
        X.append(x_i)
        Y.append(y_i)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def train(model, batch_count=100):
    def generator():
        try:
            while True:
                yield get_batch()
        except Exception as e:
            print("Exception: {}".format(e))
            pass
    print("Training start: weights avg: {}".format(model.get_weights()[0].mean()))
    model.fit_generator(generator(), samples_per_epoch=32 * 4, nb_epoch=batch_count)
    print("Training end: weights mean {}".format(model.get_weights()[0].mean()))


if __name__ == '__main__':
    output_filename = sys.argv[1]
    input_filename = None
    if len(sys.argv) > 2:
        input_filename = sys.argv[2]
    model = load_model(filename=input_filename)

    try:
        train(model)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")

    model.save(output_filename)
