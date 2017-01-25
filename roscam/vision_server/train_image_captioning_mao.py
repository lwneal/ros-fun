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
import dataset_grefexp
import image_caption
from networks import mao_net

def get_random_grefexp(reference_key=dataset_grefexp.KEY_GREFEXP_TRAIN):
    grefexp, anno, img_meta, pixels = dataset_grefexp.random_annotation(reference_key)
    x0, width, y0, height = anno['bbox']
    box = (x0, x0 + width, y0, y0 + height)
    text = random.choice(grefexp['refexps'])['raw']
    return pixels, box, text


def get_next_example():
    while True:
        pixels, box, text = get_random_grefexp()
        x = image_caption.extract_features(pixels, box)
        y = nlp_api.words_to_onehot(text, pad_to_length=image_caption.MAX_OUTPUT_WORDS)
        #print("Training on word sequence: {}".format(nlp_api.onehot_to_words(y)))
        yield x, y


def get_batch(batch_size=100):
    X = []
    Y = []
    generator = get_next_example()
    for _ in range(batch_size):
        x_i, y_i = next(generator)
        X.append(x_i)
        Y.append(y_i)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def draw_box(pixels, box):
    x0, x1, y0, y1 = box
    pixels[y0, x0:x1] = 255
    pixels[y1, x0:x1] = 255
    pixels[y0:y1, x0] = 255
    pixels[y0:y1, x1] = 255


def demonstrate(model):
    pixels, box, text = get_random_grefexp(reference_key=dataset_grefexp.KEY_GREFEXP_VAL)

    # TODO: draw pixels to screen?
    #draw_box(pixels, box)
    #jpg_data = util.encode_jpg(pixels)
    #open('/tmp/example.jpg', 'w').write(jpg_data)
    #os.system('imgcat /tmp/example.jpg')

    x = image_caption.extract_features(pixels, box)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    onehot_words = preds.reshape(preds.shape[1:])
    print("Demonstration image {}x{}".format(pixels.shape[1], pixels.shape[0]))
    print('Correct Answer: {}'.format(text))
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
    model.fit_generator(generator(), samples_per_epoch=1000, nb_epoch=1)
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
        model = mao_net.build_model()

    resnet.init()
    try:
        while True:
            train(model)
            model.save(output_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
