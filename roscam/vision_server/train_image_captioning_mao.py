#!/usr/bin/env python
import subprocess
import json
import time
import random
import socket
import os
import sys
import math

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from shared import nlp_api
import resnet
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net


def get_random_grefexp(reference_key=dataset_grefexp.KEY_GREFEXP_TRAIN):
    grefexp, anno, img_meta, jpg_data = dataset_grefexp.random_annotation(reference_key)
    x0, y0, width, height = anno['bbox']
    box = (x0, x0 + width, y0, y0 + height)
    text = random.choice(grefexp['refexps'])['raw']
    return jpg_data, box, text


def get_next_example(average_box_context=False):
    while True:
        jpg_data, box, text = get_random_grefexp()
        pixels = util.decode_jpg(jpg_data)
        preds = resnet.run(pixels)
        width, height, _ = pixels.shape
        x = image_caption.extract_features_from_preds(preds, width, height, box)
        y = nlp_api.words_to_onehot(text, pad_to_length=image_caption.MAX_OUTPUT_WORDS)
        #print("Training on word sequence: {}".format(nlp_api.onehot_to_words(y)))
        yield x, y


def get_batch(batch_size=10, **kwargs):
    X = []
    Y = []
    generator = get_next_example(**kwargs)
    for _ in range(batch_size):
        x_i, y_i = next(generator)
        X.append(x_i)
        Y.append(y_i)
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def draw_box(pixels, box):
    x0, x1, y0, y1 = box
    x1 -= 1
    y1 -= 1
    pixels[y0, x0:x1] = 255
    pixels[y1, x0:x1] = 255
    pixels[y0:y1, x0] = 255
    pixels[y0:y1, x1] = 255


def demonstrate(model):
    jpg_data, box, text = get_random_grefexp(reference_key=dataset_grefexp.KEY_GREFEXP_VAL)
    pixels = util.decode_jpg(jpg_data)

    draw_box(pixels, box)
    open('/tmp/example.jpg', 'w').write(util.encode_jpg(pixels))
    os.system('imgcat /tmp/example.jpg')

    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    x = image_caption.extract_features_from_preds(preds, width, height, box)
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    onehot_words = preds.reshape(preds.shape[1:])
    print("Demonstration image {}x{}".format(pixels.shape[1], pixels.shape[0]))
    print('Correct Answer: {}'.format(text))
    print('Prediction: {}'.format(nlp_api.onehot_to_words(onehot_words)))


def train(model, learning_rate=.05, batch_size=10):
    training_kwargs = {
        'average_box_context': False,
    }
    def generator():
        try:
            while True:
                yield get_batch(batch_size, **training_kwargs)
        except Exception as e:
            import traceback
            traceback.print_exc()


    print("Training start: weights avg: {}".format(model.get_weights()[0].mean()))
    start_time = time.time()
    history = model.fit_generator(generator(), samples_per_epoch=1000, nb_epoch=1)
    print("Training end: weights mean {}".format(model.get_weights()[0].mean()))

    info = {
            'start_time': start_time,
            'end_time': time.time(),
            'loss': history.history['loss'][0],
            'hostname': socket.gethostname(),
            'dataset': 'dataset_grefexp_train',
            'learning_rate': learning_rate,
            'optimizer': 'rmsprop',
            'batch_size': batch_size,
            'samples': 1000,
            'training_kwargs': training_kwargs,
    }
    demonstrate(model)
    return info

def save_model_info(info_filename, info, model_filename):
    if os.path.exists(info_filename):
        return
    info['model_filename'] = model_filename
    info['history'] = []
    with open(info_filename, 'w') as fp:
        fp.write(json.dumps(info, indent=2))

def get_checksum(filename):
    try:
        return subprocess.check_output(['md5sum', model_filename])
    except:
        return '0'

def save_training_info(info_filename, info, model_filename):
    info['checksum'] = get_checksum(model_filename)
    data = json.load(open(info_filename))
    data['history'].append(info)
    with open(info_filename, 'w') as fp:
        fp.write(json.dumps(data, indent=2))

if __name__ == '__main__':
    model_filename = sys.argv[1]
    if os.path.exists(model_filename):
        from keras.models import load_model
        model = load_model(model_filename)
    else:
        model = mao_net.build_model()

    info_filename = model_filename.replace('.h5', '') + '.json'
    info = {}

    save_model_info(info_filename, info, model_filename)
    resnet.init()
    # TODO: docopt or argparse
    learning_rate = float(sys.argv[2])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=learning_rate)
    model.summary()
    try:
        while True:
            train_info = train(model)
            save_training_info(info_filename, train_info, model_filename)
            model.save(model_filename)
    except KeyboardInterrupt:
        print("Stopping due to keyboard interrupt")
