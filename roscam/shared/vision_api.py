import sys
import os
import pickle
import math
import sys
import struct
import socket
from StringIO import StringIO

import capnp
import numpy as np
from PIL import Image

from shared import util
from frametalk_capnp import FrameMsg


DEFAULT_ADDR = ('127.0.0.1', 1237)


def run_resnet(jpg_data, addr=DEFAULT_ADDR):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    requestMsg = FrameMsg.new_message()
    requestMsg.frameData = jpg_data
    util.write_packet(s, requestMsg.to_bytes())

    responseMsg = util.read_packet(s)
    response_data = responseMsg['frameData']
    preds = pickle.loads(response_data)
    return preds


def detect_human(jpg_data):
    pixels = util.decode_jpg(jpg_data)

    preds = run_resnet(jpg_data)[:,:,0]

    print 'got pixels shape {} preds shape {}'.format(pixels.shape, preds.shape)
    shape = (pixels.shape[1], pixels.shape[0])
    mask = np.array(Image.fromarray(preds * 255).resize(shape)).astype(np.uint8)

    # Red overlay: output of network
    pixels[:,:,0] = mask
    return util.encode_jpg(pixels)

