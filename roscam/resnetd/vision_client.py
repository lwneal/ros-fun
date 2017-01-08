import sys
import socket
from StringIO import StringIO

import numpy as np
from PIL import Image

sys.path.append('..')
import util

DEFAULT_ADDR = ('127.0.0.1', 1237)

def resnet(jpg_data, addr=DEFAULT_ADDR):
    width, height = get_dimensions(jpg_data)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    util.write_packet(s, jpg_data)
    response_type, response_data = util.read_packet(s)
    return decode_resnet_jpg(response_data, width, height)


def decode_resnet_jpg(jpg_data, img_width, img_height):
    img = Image.open(StringIO(jpg_data))
    values = np.array(img)
    new_height = img_height / 32
    return values.reshape((new_height, -1, 2048))


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
