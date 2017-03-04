import sys
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Dropout, Merge
from keras.layers import BatchNormalization
from keras.layers import Masking
from keras.models import Model
from keras.layers import Input
from keras import layers
from keras import backend as K
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from interfaces.image_caption import MAX_OUTPUT_WORDS
from shared.nlp_api import VOCABULARY_SIZE, END_TOKEN_IDX

IMAGE_FEATURE_SIZE = 4101
MAX_WORDS = 8
WORDVEC_DIM = 100


def predict(model, x_img, x_word, timesteps=10):
    X_img = np.expand_dims(x_img, axis=0)
    X_word = np.expand_dims(x_word, axis=0)
    for _ in range(timesteps):
        preds = model.predict([X_img, X_word])
        next_word = np.argmax(preds, axis=1)
        X_word = np.concatenate([X_word, [next_word]], axis=1)
        X_img = extend(X_img)
        if next_word[0] == END_TOKEN_IDX:
            break
    return X_word[0]


def extend(X, axis=1, extend_by=1):
    shape = list(X.shape)
    shape[axis] += extend_by
    return np.resize(X, shape)


def build_model():
    # As described in https://arxiv.org/abs/1511.02283
    # Input: The 4101-dim feature from extract_features, and the previous output word

    # Hyperparameter: 
    ALPHA = 0.5

    visual_input = Sequential()
    # Embed visual down to a smaller size
    visual_input_shape = (None, None, IMAGE_FEATURE_SIZE)
    visual_input.add(TimeDistributed(Dense(
        int(WORDVEC_DIM * ALPHA),
        activation='relu',
        name='visual_embed'), batch_input_shape=visual_input_shape))

    word_input = Sequential()
    word_input.add(layers.Embedding(VOCABULARY_SIZE, WORDVEC_DIM, dropout=.5))

    model = Sequential()
    model.add(Merge([visual_input, word_input], mode='concat', concat_axis=2))

    model.add(LSTM(1024, name='lstm_1', return_sequences=False))
    model.add(layers.Dropout(.5))

    model.add(Dense(
        VOCABULARY_SIZE,
        activation='softmax',
        name='embed_out'))

    return model
