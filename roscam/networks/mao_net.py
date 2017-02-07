import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from interfaces.image_caption import MAX_OUTPUT_WORDS
from shared.nlp_api import VOCABULARY_SIZE

BATCH_SIZE=16
IMAGE_FEATURE_SIZE = 4101


def build_model():
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, TimeDistributed, Activation, Dropout, Merge
    from keras.layers import BatchNormalization
    from keras.layers import Masking

    # As described in https://arxiv.org/abs/1511.02283
    # Input: The 4101-dim feature from extract_features, and the previous output word

    visual_input = Sequential()
    visual_input.add(BatchNormalization(batch_input_shape=(BATCH_SIZE, 1, IMAGE_FEATURE_SIZE)))

    word_input = Sequential()
    word_input_shape=(BATCH_SIZE, 1, VOCABULARY_SIZE)
    word_input.add(Masking(batch_input_shape=word_input_shape))

    model = Sequential()
    model.add(Merge([visual_input, word_input], mode='concat', concat_axis=2))
    model.add(LSTM(1024, name='lstm_1', return_sequences=True, stateful=True))
    model.add(TimeDistributed(Dense(VOCABULARY_SIZE, name='fc_1')))
    model.add(Activation('softmax', name='softmax_1'))
    return model
