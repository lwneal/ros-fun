import os
import cPickle
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
import sys
from scipy.misc import imread, imresize


input_width = 224
input_height = 224

# TODO: image path
def VGG_16(weights_path, image_path='static/kinect.jpg'):
    from keras.models import Sequential
    from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
    from keras import backend as K
    import h5py

    input_tensor = K.placeholder((1, 3, input_width, input_height))

    # build the VGG16 network with our 3 images as input
    first_layer = ZeroPadding2D((1, 1))
    first_layer.set_input(input_tensor, shape=(1, 3, input_width, input_height))

    model = Sequential()
    model.add(first_layer)
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

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

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    eval_function = K.function([input_tensor], outputs_dict)
    return eval_function


def load_cached_model(filename='imagenet.pkl'):
    if os.path.isfile(filename):
        with open(filename) as fp:
            return cPickle.load(fp)
    return None


def cache_model(model, filename='imagenet.pkl'):
    with open(filename, 'w') as fp:
        cPickle.dump(model, fp)


def convert_image(filename):
    image = load_square_image(filename)
    width, height, channels = image.shape
    assert width == height
    assert channels == 3
    im = imresize(image, (input_width, input_height)).astype(np.float32)
    # TODO: What are these magic numbers?
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    # TODO: Is BGR conversion correct here?
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    return im


def load_square_image(filename):
    im = to_rgb(imread(filename))
    width, height, channels = im.shape
    # Crop to the largest centered square
    if width > height:
        left = (width / 2) - height/2
        right = left + height
        return im[left:right, :]
    else:
        top = (height / 2) - width/2
        bottom = top + width
        return im[:, top:bottom]


def to_rgb(im):
    # Monochrome -> RGB
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]
        im = np.concatenate((im, im, im), axis=2)
    return im
