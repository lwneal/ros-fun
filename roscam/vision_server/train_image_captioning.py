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
import spatial_context_net


def bbox(region):
    return (region['x'], region['x'] + region['width'], region['y'], region['y'] + region['height'])


def get_next_example():
    while True:
        meta, pixels = dataset_coco.random_image()
        preds = resnet.run(pixels)
        #for region in meta['regions']:
        phrase = random.choice(region['phrase'])
        phrase = '{} {} {}'.format(nlp_api.START_TOKEN, phrase[:spatial_context_net.MAX_OUTPUT_WORDS-2], nlp_api.END_TOKEN)
        print phrase
        x0, x1, y0, y1 = [v/32 for v in bbox(region)]  # Convert from image to resnet output coordinates
        x = spatial_context_net.extract_features(pixels, (x0+x1)/2, (y0+y1)/2, preds=preds)
        y = nlp_api.words_to_onehot(phrase, pad_to_length=spatial_context_net.MAX_OUTPUT_WORDS)
        yield x, y


def get_batch(batch_size=20):
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
    try:
        draw_box(pixels, box)
    except:
        pass

    open('/tmp/example.jpg', 'w').write(util.encode_jpg(pixels))
    os.system('imgcat /tmp/example.jpg')

    x = spatial_context_net.extract_features(pixels, box[0]/32, box[2]/32)
    x = np.expand_dims(x, axis=0)
    x = [x.squeeze(axis=-1) for x in np.split(x, 4, axis=-1)]
    preds = model.predict(x)
    onehot_words = preds.reshape(preds.shape[1:])
    print('Prediction: {}'.format(nlp_api.onehot_to_words(onehot_words)))


def train(model):
    def generator():
        try:
            while True:
                yield get_batch()
        except Exception as e:
            import traceback
            traceback.print_exc()
    print("Training start: weights avg: {}".format(model.get_weights()[0].mean()))
    model.fit_generator(generator(), samples_per_epoch=256, nb_epoch=1)
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
