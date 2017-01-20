#!/usr/bin/env python
import random
import os
import sys
import math

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from shared import nlp_api
import resnet
import dataset_coco
import baseline_net as network


def bbox(region):
    return (region['x'], region['x'] + region['width'], region['y'], region['y'] + region['height'])

def draw_box(pixels, box):
    x0, x1, y0, y1 = box
    pixels[y0, x0:x1] = 255
    pixels[y1, x0:x1] = 255
    pixels[y0:y1, x0] = 255
    pixels[y0:y1, x1] = 255


def demonstrate(model):
    meta, pixels = dataset_coco.random_image()

    open('/tmp/example.jpg', 'w').write(util.encode_jpg(pixels))
    os.system('imgcat /tmp/example.jpg')

    width, height = meta['width'], meta['height']
    x = network.extract_features(pixels, height/2, width/2)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    onehot_words = preds.reshape(preds.shape[1:])
    print('Prediction: {}'.format(nlp_api.onehot_to_words(onehot_words)))


if __name__ == '__main__':
    input_filename = sys.argv[1]

    from keras.models import load_model
    model = load_model(input_filename)
    demonstrate(model)
