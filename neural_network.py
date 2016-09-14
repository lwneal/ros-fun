import os
import numpy as np
import random
import h5py

VGG_WEIGHTS_FILE = 'vgg16_weights.h5'
IMG_INPUT_WIDTH = 324
IMG_INPUT_HEIGHT = 324
OUTPUT_SHAPE = (20, 20)
BATCH_OUTPUT_SHAPE = (1, 1, 20, 20)


def build_vgg_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout, Merge
    from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten
    from keras import backend as K
    import h5py

    vgg_model = Sequential()
    vgg_model.add(ZeroPadding2D((1,1), input_shape=(3, IMG_INPUT_WIDTH, IMG_INPUT_HEIGHT)))
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    vgg_model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    vgg_model.add(ZeroPadding2D((1, 1)))
    vgg_model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))

    load_vgg16_conv_weights(vgg_model)
    #vgg_model.compile(loss='categorical_crossentropy', optimizer='sgd')

    # Compress feature map down to a more manageable size
    vgg_model.add(Convolution2D(32, 1, 1, activation='relu', name='conv6'))
    return vgg_model


def build_model(wordvec_dim, sequence_length):
    from keras.models import Sequential
    from keras.layers import Convolution2D, ZeroPadding2D
    from keras.layers.core import Dense, Activation, Dropout, Merge, Flatten, RepeatVector, Reshape
    from keras.layers.recurrent import LSTM, GRU

    vgg_model = build_vgg_model()

    language_model = Sequential()
    language_model.add(GRU(512, input_shape=(sequence_length, wordvec_dim), name="gru_1"))
    # Broadcast fixed-size language output to variable-size convnet
    language_model.add(RepeatVector(20*20))
    language_model.add(Reshape((512,20,20)))

    model = Sequential()
    model.add(Merge([vgg_model, language_model], mode='concat', concat_axis=1))

    # Add another layer to combine vision and language
    # Note that the number of vision and language outputs are roughly balanced
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name="fusion_conv1"))

    # Another layer for more logic
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(64, 5, 5, activation='relu', name="fusion_conv2"))

    # Another layer for more logic
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(64, 5, 5, activation='relu', name="fusion_conv3"))

    # Another another layer for more more logic
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(64, 5, 5, activation='relu', name="fusion_conv4"))

    # Final layer
    model.add(Convolution2D(1, 1, 1, activation='sigmoid', name="output"))

    from keras.optimizers import RMSprop, Adam
    model.compile(loss='mse', optimizer=RMSprop(lr=.00002))
    return model


def load_vgg16_conv_weights(model, weights_path=VGG_WEIGHTS_FILE):
    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    # note: when there is a complete match between your model definition
    # and your weight savefile, you can simply call model.load_weights(filename)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')

