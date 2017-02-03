import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from interfaces.image_caption import MAX_OUTPUT_WORDS, VOCABULARY_SIZE

BATCH_SIZE=16


def build_model():
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed, Activation, Dropout

    # As described in https://arxiv.org/abs/1511.02283
    # Input: The 4101-dim feature from extract_features, and the previous output word
    model = Sequential()
    input_shape = (BATCH_SIZE, 1, 4101 + VOCABULARY_SIZE)
    model.add(LSTM(1024, batch_input_shape=input_shape, name='lstm_1', stateful=True))
    model.add(Dense(VOCABULARY_SIZE, name='fc_2'))
    model.add(Activation('softmax', name='softmax_1'))
    return model
