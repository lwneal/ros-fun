from cStringIO import StringIO
import numpy as np
from PIL import Image
from scipy.misc import imresize

import neural_network

neural_network.init()
preds = neural_network.run(np.zeros((256, 256, 3)))

"""
Entry point for computer vision.
Input: A JPG image video frame
Output: A JPG image with labels and annotations
"""
def computer_vision(jpg_data):
    pixels = decode_jpg(jpg_data)

    preds = neural_network.run(pixels)

    # Display a few neurons
    for idx in range(10):
        display_neuron = imresize(preds[0][idx], (64, 64))
        x0 = 64 * idx
        pixels[:64, x0:x0+64] = 0
        for c in range(3):
            pixels[:64, x0:x0+64, c] = display_neuron

    return encode_jpg(pixels)

def decode_jpg(jpg_data):
    fp = StringIO(jpg_data)
    img = Image.open(fp)
    return np.array(img)

def encode_jpg(pixels):
    pil_img = Image.fromarray(pixels)
    fp = StringIO()
    pil_img.save(fp, format='JPEG')
    return fp.getvalue()

