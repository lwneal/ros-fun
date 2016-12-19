from cStringIO import StringIO
import numpy as np
from PIL import Image

import neural_network

neural_network.init()

"""
Entry point for computer vision.
Input: A raw JPG image
Output: A JPG image with labels and annotations
"""
def computer_vision(jpg_data):
    pixels = decode_jpg(jpg_data)

    output = neural_network.run(pixels)

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

