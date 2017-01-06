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
    preds.reshape((2048, -1))
    return encode_jpg(preds)

def decode_jpg(jpg_data):
    fp = StringIO(jpg_data)
    img = Image.open(fp)
    return np.array(img)

def encode_jpg(pixels):
    pil_img = Image.fromarray(pixels.astype(np.uint8))
    fp = StringIO()
    pil_img.save(fp, format='JPEG')
    return fp.getvalue()

