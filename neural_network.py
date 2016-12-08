import os
import numpy as np
import random
import h5py

VGG_WEIGHTS_FILE = 'vgg16_conv_weights.h5'
OUTPUT_SHAPE = (24, 24)
BATCH_OUTPUT_SHAPE = (1, 1, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])
IMG_INPUT_HEIGHT = OUTPUT_SHAPE[0] * 16
IMG_INPUT_WIDTH = OUTPUT_SHAPE[1] * 16


def build_vgg_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation, Dropout, Merge
    from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten
    from keras import backend as K
    import h5py

    vgg_model = Sequential()
    vgg_model.add(ZeroPadding2D((1,1), input_shape=(3, IMG_INPUT_HEIGHT, IMG_INPUT_WIDTH)))
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

    #load_weights(vgg_model, VGG_WEIGHTS_FILE)

    # Compress feature map down to a more manageable size
    vgg_model.add(Convolution2D(128, 1, 1, activation='relu', name='conv6'))
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
    language_model.add(RepeatVector(OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]))
    language_model.add(Reshape((512, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))

    model = Sequential()
    for layer in vgg_model.layers:
        print("Layer {}: weights shape: {}".format(
            layer, [w.shape for w in layer.get_weights()]))
    vgg_model.add(Reshape((128, OUTPUT_SHAPE[0], OUTPUT_SHAPE[1])))
    model.add(Merge([vgg_model, language_model], mode='concat', concat_axis=1))

    # Add another layer to combine vision and language
    # Note that the number of vision and language outputs are roughly balanced
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name="fusion_conv1"))
    model.add(Dropout(0.2))

    # Another layer for more logic
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(128, 5, 5, activation='relu', name="fusion_conv2"))
    model.add(Dropout(0.2))

    # more layers
    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(128, 5, 5, activation='relu', name="fusion_conv3"))
    model.add(Dropout(0.2))

    # Final layer
    model.add(Convolution2D(1, 1, 1, activation='sigmoid', name="output"))

    from keras.optimizers import RMSprop, Adam
    model.compile(loss='mse', optimizer=RMSprop(lr=.00002))
    return model


# Load weights from an HDF5 file into a Keras model
# Requires that each layer in the model have a unique name
def load_weights(model, filename):
    import time
    import h5py
    start_time = time.time()
    print("Loading weights from filename {} to model {}".format(filename, model))
    f = h5py.File(filename)

    # Collect nested layers (eg. layers inside a Merge)
    model_matrices = {}
    def collect_layers(item):
        for layer in item.layers:
            if hasattr(layer, 'layers'):
                collect_layers(layer)
            for mat in layer.weights:
                if mat.name in model_matrices and model_matrices[mat.name] is not mat:
                    print("Warning: Found more than one layer named {} in model {}".format(mat.name, model))
                model_matrices[mat.name] = mat
    collect_layers(model)

    # Load matrices, discarding extra weights or padding with zeros as required
    loaded_count = 0
    for layer_name in f:
        saved_layer = f[layer_name]
        for mat_name in saved_layer:
            saved_mat = saved_layer[mat_name]
            if mat_name not in model_matrices:
                print("Warning: Discarding unknown matrix {}".format(mat_name))
                continue
            mat_shape = model_matrices[mat_name].get_value().shape
            if mat_shape != saved_mat.shape:
                print("Layer {} resizing saved matrix {} to new shape {}".format(
                    mat_name, mat_shape, saved_mat.shape))
                saved_mat.value.resize(mat_shape)
            model_matrices[mat_name].set_value(saved_mat.value)
            loaded_count += 1
    print("Loaded {} matrices in {:.2f}s".format(loaded_count, time.time() - start_time))
    f.close()
