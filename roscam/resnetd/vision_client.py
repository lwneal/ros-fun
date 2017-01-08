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
    height = struct.unpack('!l', response_data[0:4])[0]
    width = struct.unpack('!l', response_data[4:8])[0]
    jpg_data = response_data[8:]
    return decode_resnet_jpg(height, width, jpg_data)


def decode_resnet_jpg(height, width, jpg_data):
    img = Image.open(StringIO(jpg_data))
    values = np.array(img)
    return values.reshape((height, width, 2048))


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
