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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from interfaces.image_caption import MAX_OUTPUT_WORDS
from shared.nlp_api import VOCABULARY_SIZE

BATCH_SIZE=16
IMAGE_FEATURE_SIZE = 4101


class TiedDense(layers.Dense):
    def __init__(self, output_dim, master_layer, init='glorot_uniform', activation='linear', weights=None,
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None, input_dim=None, **kwargs):
        self.master_layer = master_layer
        super(TiedDense, self).__init__(output_dim, **kwargs)

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


def build_model():
    # As described in https://arxiv.org/abs/1511.02283
    # Input: The 4101-dim feature from extract_features, and the previous output word

    visual_input = Sequential()
    visual_input.add(BatchNormalization(batch_input_shape=(BATCH_SIZE, 1, IMAGE_FEATURE_SIZE), name='batch_norm_1'))

    dummy_model = Sequential()
    unembed_layer = Dense(VOCABULARY_SIZE, input_shape=(1024,), name='embed_output')
    dummy_model.add(unembed_layer)
    embed_layer = TiedDense(1024, master_layer=unembed_layer, name='embed_input')

    word_input = Sequential()
    word_input_shape=(BATCH_SIZE, 1, VOCABULARY_SIZE)
    word_input.add(Masking(batch_input_shape=word_input_shape))
    word_input.add(TimeDistributed(embed_layer))
    word_input.add(Dropout(0.5))

    model = Sequential()
    model.add(Merge([visual_input, word_input], mode='concat', concat_axis=2))
    model.add(LSTM(1024, name='lstm_1', return_sequences=True, stateful=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(unembed_layer))
    model.add(Activation('softmax', name='softmax_1'))

    return model
