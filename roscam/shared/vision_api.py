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
import capnp
from frametalk_capnp import FrameMsg


DEFAULT_ADDR = ('127.0.0.1', 1237)


def resnet(jpg_data, addr=DEFAULT_ADDR):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    requestMsg = FrameMsg.new_message()
    requestMsg.frameData = jpg_data
    util.write_packet(s, requestMsg.to_bytes())

    responseMsg = util.read_packet(s)
    response_data = responseMsg['frameData']
    preds = pickle.loads(response_data)
    return preds


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
