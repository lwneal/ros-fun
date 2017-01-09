import pickle
import math
import sys
import struct
import socket
from StringIO import StringIO

import numpy as np
from PIL import Image

sys.path.append('..')
import util

DEFAULT_ADDR = ('127.0.0.1', 1237)

def resnet(jpg_data, addr=DEFAULT_ADDR):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    util.write_packet(s, jpg_data)
    response_type, response_data = util.read_packet(s)
    preds = pickle.loads(response_data)
    return preds


def decode_resnet_jpg(height, width, jpg_data):
    img = Image.open(StringIO(jpg_data))
    values = np.array(img)
    preds = values.reshape((width, height, 2048))
    preds = np.transpose(preds, (1, 0, 2))
    print
    print preds.shape
    print preds[:,:,42]
    return preds


def get_dimensions(jpg_data):
    img = Image.open(StringIO(jpg_data))
    return img.size


if __name__ == '__main__':
    addr = DEFAULT_ADDR
    if len(sys.argv) < 2:
        print("Usage: {} input.jpg [server] > output.jpg".format(sys.argv[0]))
        print("server: defaults to localhost")
        exit()
    jpg_data = open(sys.argv[1]).read()
    if len(sys.argv) > 2:
        addr = (sys.argv[2], 1237)
    activations = resnet(jpg_data, addr)
    print("Activations shape {}".format(activations.shape))
    print(activations)
