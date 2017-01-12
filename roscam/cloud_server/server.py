import time
import socket
import struct
import sys
import os

import capnp
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from shared import vision_api
from cloud_server import block_storage
from frametalk_capnp import FrameMsg


def handle_robot(robot_sock, subscriber_sock):
    bs = block_storage.BlockStorageContext()
    while True:
        msg = util.read_packet(robot_sock)
        frame_jpg = msg['frameData']
        timestamp = msg['timestampEpoch']

        bs.store(frame_jpg, timestamp)

        preds = vision_api.detect_human(frame_jpg)

        annotated_jpg = build_detection_visualization(frame_jpg, preds)

        outputMsg = FrameMsg.new_message()
        outputMsg.frameData = annotated_jpg
        outputMsg.timestampEpoch = timestamp
        util.write_packet(subscriber_sock, outputMsg.to_bytes())


def build_detection_visualization(frame_jpg, preds):
    pixels = util.decode_jpg(frame_jpg)

    # TODO: Get rid of imresize
    from scipy.misc import imresize
    preds = imresize(preds, pixels.shape)

    # Set the red channel to the output of the detector
    pixels[:,:,0] = 0.5 * pixels[:,:,0] + 0.5 * preds
    return util.encode_jpg(pixels)


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 1234))
    s.listen(1)

    t = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    t.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    t.bind(('0.0.0.0', 1235))
    t.listen(1)

    while True:
        print("Waiting for connection")

        def connect(s):
            conn, addr = s.accept()
            return conn

        print("Waiting for ROS client connection")
        conn = connect(s)
        print("Waiting for website viewer connection")
        conn2 = connect(t)
        handle_robot(conn, conn2)

    sys.stderr.write("Connection closed\n")
