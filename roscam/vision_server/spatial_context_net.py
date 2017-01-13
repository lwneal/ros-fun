# A toy task for learning spatial context for image caption output

# First we build the network

# Input is four variable-length sequences of ResNet features
# An encoder transforms those to a fixed-size representation
# Then a decoder generates a sequence of output words
import time

import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import util
import random

from keras.layers.core import Merge
from keras.models import *
from keras.layers import *

import resnet
resnet.init()

TIMESTEPS = 16
MAX_OUTPUT_WORDS = 32

def load_img(name):
    return util.decode_jpg(open(name).read())

def save_img(name, img):
    open(name, 'w').write(util.encode_jpg(img))

print("Loading test images")
cat = load_img('animals/kitten.jpg')
dog = load_img('animals/puppy.jpg')
beaver = load_img('animals/beaver.jpg')


def str_to_onehot(words):
    y = np.zeros((MAX_OUTPUT_WORDS, 128))
    for i in range(len(words)):
        idx = ord(words[i])
        y[i][idx] = 1.0
    # Fill the rest with space characters
    for i in range(len(words), MAX_OUTPUT_WORDS):
        y[i,ord(' ')] = 1.0
    return y


def onehot_to_str(preds):
    output = ""
    for letter in preds:
        output += chr(np.argmax(letter))
    return output


def insert_animal(image, animal, x, y):
    animal_height = animal.shape[0]
    animal_width = animal.shape[1]
    image[x:x + animal_width, y:y + animal_height, :] = animal

def get_animal():
    return random.choice([
        ('cat', cat),
        ('dog', dog),
        ('beaver', beaver),
    ])

def make_animal_image():
    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    def rand(maxval):
        return np.random.randint(1 + maxval / 32) * 32
    # Put an animal into the image
    mx = 1024 - 256  # Don't hit the edges
    x, y = rand(mx), rand(mx)
    first_animal_name, first_animal_img = get_animal()
    insert_animal(img, first_animal_img, x, y)
    # Put another animal into the image
    second_animal_name, second_animal_img = get_animal()
    direction_phrase, x2, y2 = random.choice([
        ('left of', rand(x), y),
        ('right of', x + rand(mx - x), y),
        ('above', x, rand(y)),
        ('below', x, y + rand(mx - y))
    ])
    insert_animal(img, second_animal_img, x2, y2)
    training_bbox = (x2, x2 + 256, y2, y2 + 256)
    training_phrase = ("{} {} {}".format(first_animal_name, direction_phrase, second_animal_name))
    return img, training_bbox, training_phrase

def left_pad(input_array, to_size=TIMESTEPS):
    input_size, feature_dim = input_array.shape
    y = np.zeros((to_size, feature_dim))
    if input_size > 0:
        y[-input_size:] = input_array[:to_size]
    return y

def clip(x, minval=0, maxval=32):
    return np.clip(x, minval, maxval)

def pad_preds(preds, to_size=32):
    height, width, depth = preds.shape
    height = min(height, to_size)
    width = min(width, to_size)
    padded = np.zeros((to_size, to_size, depth))
    padded[:height, :width] = preds[:height, :width]
    return padded

def extract_features(img, x, y, preds=None):
    example_x = np.zeros((TIMESTEPS, 2048, 4))
    resnet_preds = preds if preds is not None else resnet.run(img)
    # Pad preds to 32x32
    resnet_preds = pad_preds(resnet_preds, to_size=32)
    x = np.clip(x, 0, 31)
    y = np.clip(y, 0, 31)
    # Extract context, padded and in correct order, from left/right/top/bottom
    example_x[:,:,0] = left_pad(resnet_preds[y,                         clip(x - TIMESTEPS) : x,    :])
    example_x[:,:,1] = left_pad(resnet_preds[y,                         x : clip(x + TIMESTEPS),    :][::-1])
    example_x[:,:,2] = left_pad(resnet_preds[clip(y - TIMESTEPS) : y,   x,                          :])
    example_x[:,:,3] = left_pad(resnet_preds[y : clip(y + TIMESTEPS),   x,                          :][::-1])
    return example_x

def extract_example(img, box, phrase, preds=None):
    # Convert from image coordinates to resnet activation coordinates
    x0, x1, y0, y1 = (val/32 for val in box)
    x, y = (x0 + x1) / 2, (y0+y1)/2
    example_x = extract_features(img, x, y, preds=preds)
    # Expected output: the desired phrase
    example_y = np.zeros((MAX_OUTPUT_WORDS, 128))
    letters = str_to_onehot(phrase)
    example_y[:len(letters)] = letters
    return example_x, example_y

def get_training(training_size=100):
    X = np.zeros((training_size, TIMESTEPS, 2048, 4))
    Y = np.zeros((training_size, MAX_OUTPUT_WORDS, 128))
    for i in range(training_size):
        img, box, phrase = make_animal_image()
        x, y = extract_example(img, box, phrase)
        X[i] = x
        Y[i] = y
    xvals = [x.squeeze(axis=-1) for x in np.split(X, 4, axis=-1)]
    return xvals, Y


def train_one_round(model):
    # Do some training
    X, Y = get_training(training_size=256)
    model.fit(X, Y, nb_epoch=10)
    model.save('model.h5')

def demonstrate(model):
    # Demonstrate on an image
    img, box, correct_phrase = make_animal_image()
    save_img('/tmp/animals.jpg', img)
    os.system('imgcat /tmp/animals.jpg')
    x = extract_features(img, box[0]/32, box[2]/32)
    x = np.expand_dims(x, axis=0)
    x = [x.squeeze(axis=-1) for x in np.split(x, 4, axis=-1)]
    preds = model.predict(x)
    print('Prediction: {}'.format(onehot_to_str(preds[0])))


def spatialContextNet():
    model = Sequential()
    input_shape = (None, 2048)
    gru = GRU(512, input_shape=input_shape, return_sequences=False)
    model.add(gru)
    model.add(Dense(512, activation='sigmoid'))
    return model

def build_model():
    left = spatialContextNet()
    right = spatialContextNet()
    up = spatialContextNet()
    down = spatialContextNet()

    model = Sequential()
    model.add(Merge([left, right, up, down], mode='concat', concat_axis=1))
    model.add(Dense(512, activation='relu'))
    model.add(RepeatVector(MAX_OUTPUT_WORDS))
    model.add(GRU(128, return_sequences=True))
    model.add(TimeDistributed(Dense(128)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', learning_rate=1.0)
    return model

if __name__ == '__main__':
    model = build_model()
    while True:
        train_one_round(model)
        for i in range(10):
            demonstrate(model)
