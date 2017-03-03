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
from shared.nlp_api import VOCABULARY_SIZE

BATCH_SIZE=16
IMAGE_FEATURE_SIZE = 4101
MAX_WORDS = 8
WORDVEC_DIM = 100


class TiedTransposeDense(layers.Dense):
    def __init__(self, output_dim, master_layer, **kwargs):
        self.master_layer = master_layer
        super(TiedTransposeDense, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.input_dim = input_dim

        self.W = tf.transpose(self.master_layer.W)
        self.b = K.zeros((self.output_dim,))
        self.params = [self.b]
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True


from nlp_server import word_vector
def load_glove_weights():
    word_vector.init()
    glove_weights = np.zeros((WORDVEC_DIM, VOCABULARY_SIZE))
    for i in range(VOCABULARY_SIZE):
        glove_weights[:, i] = word_vector.glove_dict[word_vector.idx_word[i]]
    return glove_weights


def build_model():
    # As described in https://arxiv.org/abs/1511.02283
    # Input: The 4101-dim feature from extract_features, and the previous output word

    # Hyperparameter: 
    ALPHA = 0.5

    visual_input = Sequential()
    # Embed visual down to a smaller size
    visual_input_shape = (None, None, IMAGE_FEATURE_SIZE)
    visual_input.add(layers.BatchNormalization(batch_input_shape=visual_input_shape))
    visual_input.add(TimeDistributed(Dense(
        int(WORDVEC_DIM * ALPHA),
        activation='relu',
        name='visual_embed')))
    visual_input.add(layers.BatchNormalization())

    word_input = Sequential()
    word_input.add(layers.Embedding(VOCABULARY_SIZE, WORDVEC_DIM, dropout=.5))
    word_input.add(layers.BatchNormalization())

    model = Sequential()
    model.add(Merge([visual_input, word_input], mode='concat', concat_axis=2))
    model.add(layers.BatchNormalization())

    model.add(LSTM(1024, name='lstm_1', return_sequences=False))
    model.add(layers.Dropout(.5))
    model.add(layers.BatchNormalization())

    model.add(Dense(
        VOCABULARY_SIZE,
        activation='softmax',
        name='embed_out'))

    return model
