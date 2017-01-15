#!/usr/bin/env python
import random
import os
import sys
import math

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
import resnet
import dataset_coco
import spatial_context_net


import resnet
resnet.init()

def bbox(region):
    return (region['x'], region['x'] + region['width'], region['y'], region['y'] + region['height'])

def get_next_example():
    while True:
        meta, pixels = dataset_coco.random_image()
        preds = resnet.run(pixels)
        for region in meta['regions']:
            phrase = region['phrase']
            phrase = phrase[:spatial_context_net.MAX_OUTPUT_WORDS]
            box = bbox(region)
            x, y = spatial_context_net.extract_example(pixels, box, phrase, preds=preds)
            yield x, y


def get_batch(batch_size=32):
    X = []
    Y = []
    generator = get_next_example()
    for _ in range(batch_size):
        x_i, y_i = next(generator)
        X.append(x_i)
        Y.append(y_i)
    X = np.array(X)
    Y = np.array(Y)
    xvals = [x.squeeze(axis=-1) for x in np.split(X, 4, axis=-1)]
    return xvals, Y


def draw_box(pixels, box):
    x0, x1, y0, y1 = box
    pixels[y0, x0:x1] = 255
    pixels[y1, x0:x1] = 255
    pixels[y0:y1, x0] = 255
    pixels[y0:y1, x1] = 255


def demonstrate(model):
    meta, pixels = dataset_coco.random_image()
    region = random.choice(meta['regions'])
    box = bbox(region)
    draw_box(pixels, box)

    open('/tmp/example.jpg', 'w').write(util.encode_jpg(pixels))
    os.system('imgcat /tmp/example.jpg')

    x = spatial_context_net.extract_features(pixels, box[0]/32, box[2]/32)
    x = np.expand_dims(x, axis=0)
    x = [x.squeeze(axis=-1) for x in np.split(x, 4, axis=-1)]
    preds = model.predict(x)
    print('Prediction: {}'.format(spatial_context_net.onehot_to_str(preds[0])))


def train(model, batch_count=8):
    def generator():
        try:
            while True:
                yield get_batch()
        except Exception as e:
            import traceback
            traceback.print_exc()
    print("Training start: weights avg: {}".format(model.get_weights()[0].mean()))
    model.fit_generator(generator(), samples_per_epoch=1024, nb_epoch=batch_count)
    print("Training end: weights mean {}".format(model.get_weights()[0].mean()))
    demonstrate(model)


if __name__ == '__main__':
    output_filename = sys.argv[1]
    input_filename = None
    if len(sys.argv) > 2:
        input_filename = sys.argv[2]
        from keras.models import load_model
        model = load_model(input_filename)
    else:
        model = spatial_context_net.build_model()

    try:
        while True:
            train(model)
            model.save(output_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
