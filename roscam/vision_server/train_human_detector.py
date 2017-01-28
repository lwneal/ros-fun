#!/usr/bin/env python
import os
import sys
import math

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
import resnet
from interfaces import image_salience
from datasets import dataset_coco
from networks import conv_detector_net


def get_example():
    meta, pixels = dataset_coco.random_image()
    mask = dataset_coco.human_detection_mask(meta)

    x = resnet.run(pixels)
    y = util.rescale(mask, x.shape)

    # HACK: Fill all images into a max-sized 32x32 activation mask
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


def train(model, output_filename, batch_count=100):
    def generator():
        try:
            while True:
                yield get_batch()
        except Exception as e:
            import traceback
            traceback.print_exc()
    print("Training start: weights avg: {}".format(model.get_weights()[0].mean()))
    model.fit_generator(generator(), samples_per_epoch=32 * 4, nb_epoch=batch_count)
    print("Training end: weights mean {}".format(model.get_weights()[0].mean()))
    model.save(output_filename)


if __name__ == '__main__':
    output_filename = sys.argv[1]
    input_filename = None
    if len(sys.argv) > 2:
        input_filename = sys.argv[2]
        model = load_model(input_filename)
    else:
        model = conv_detector_net.build_model()

    resnet.init()
    try:
        train(model, output_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
