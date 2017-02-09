import sys
import zlib
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
from frametalk_capnp import FrameMsg, VisionRequestType


DEFAULT_ADDR = ('localhost', 1237)


def vision_request(jpg_data, request_type, addr=DEFAULT_ADDR):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    requestMsg = FrameMsg.new_message()
    requestMsg.frameData = jpg_data
    requestMsg.visionType = request_type
    util.write_packet(s, requestMsg.to_bytes())

    responseMsg = util.read_packet(s)
    return responseMsg


def run_resnet(jpg_data):
    responseMsg = vision_request(jpg_data, request_type=VisionRequestType.resNet50)
    preds = pickle.loads(zlib.decompress(responseMsg['frameData']))
    return preds


def detect_human(jpg_data):
    responseMsg = vision_request(jpg_data, request_type=VisionRequestType.detectHuman)
    preds = pickle.loads(zlib.decompress(responseMsg['frameData']))
    command = responseMsg['robotCommand']
    return preds, command

