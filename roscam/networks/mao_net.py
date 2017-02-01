import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from interfaces.image_caption import MAX_OUTPUT_WORDS, VOCABULARY_SIZE


def build_model():
    from keras.models import *
    from keras.layers import *

    # As described in https://arxiv.org/abs/1511.02283
    # Input: The 4101-dim feature from extract_features
    model = Sequential()
    model.add(RepeatVector(MAX_OUTPUT_WORDS, input_shape=(4101,)))
    model.add(LSTM(1024, name='lstm_1', return_sequences=True))
    model.add(TimeDistributed(Dense(1024, name='fc_1')))
    model.add(TimeDistributed(Dense(VOCABULARY_SIZE, name='fc_2')))

    # Output: Prediction among all possible words
    model.add(Activation('softmax', name='softmax_1'))
    return model
