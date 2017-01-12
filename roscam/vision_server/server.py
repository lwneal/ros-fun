import pickle
import time
import sys
import struct
import os
import socket
from cStringIO import StringIO

import numpy as np
from PIL import Image
import capnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resnet
from shared import util
from frametalk_capnp import FrameMsg, VisionRequestType


def resnet_jpg(pixels):
    preds = resnet.run(pixels)
    # Remove extra dimension
    preds = preds.reshape(preds.shape[1:])
    # Round activations to 8-bit values
    preds = preds.astype(np.uint8)
    return preds


def detect_human_request(pixels):
    preds = human_detector.run(pixels)
    # Remove extra dimension
    preds = preds.reshape(preds.shape[1:])
    # Round activations to 8-bit values
    preds = preds.astype(np.uint8)
    return preds


def handle_client(conn):
    start_time = time.time()
    msg = util.read_packet(conn)
    jpg_image = msg['frameData']
    requestType = msg['visionType']

    pixels = util.decode_jpg(jpg_image)

    if requestType == 0:
        # Return rounded preds as pickle
        preds = resnet_request(pixels)
    elif requestType == VisionRequestType.detectHuman:
        # Detect humans, return pickled preds
        preds = detect_human_request(pixels)

    outputMsg = FrameMsg.new_message()
    outputMsg.frameData = pickle.dumps(preds)
    util.write_packet(conn, outputMsg.to_bytes())
    print("Finished request type {} in {:.3f}s".format(requestType, time.time() - start_time))


if __name__ == '__main__':

    resnet.init()
    human_detector.init()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 1237))
    s.listen(1)

    while True:
        #print("Waiting for connection...")
        conn, addr = s.accept()
        start_time = time.time()
        handle_client(conn)
