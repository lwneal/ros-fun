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
from frametalk_capnp import FrameMsg, VisionRequestType


DEFAULT_ADDR = ('127.0.0.1', 1237)


def vision_request(jpg_data, request_type, addr=DEFAULT_ADDR):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(addr)
    requestMsg = FrameMsg.new_message()
    requestMsg.frameData = jpg_data
    requestMsg.visionType = request_type
    util.write_packet(s, requestMsg.to_bytes())

    responseMsg = util.read_packet(s)
    response_data = responseMsg['frameData']
    preds = pickle.loads(response_data)
    return preds


def run_resnet(jpg_data):
    return vision_request(jpg_data, request_type=VisionRequestType.resNet50)


def detect_human(jpg_data):
    return vision_request(jpg_data, request_type=VisionRequestType.detectHuman)
