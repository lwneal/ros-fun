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


MODELS_DIR = '/home/nealla/models'


def build_visual_output(pixels, preds):
    print 'got pixels shape {} preds shape {}'.format(pixels.shape, preds.shape)
    shape = (pixels.shape[1], pixels.shape[0])
    mask = np.array(Image.fromarray(preds * 255).resize(shape)).astype(np.uint8)

    # Red overlay: output of network
    pixels[:,:,0] = mask
    return util.encode_jpg(pixels)


def handle_robot(robot_sock, subscriber_sock):
    bs = block_storage.BlockStorageContext()
    while True:
        msg = util.read_packet(robot_sock)
        frame_jpg = msg['frameData']
        timestamp = msg['timestampEpoch']

        bs.store(frame_jpg, timestamp)  # TODO: propagate timestamps from ROS using capnp

        #preds = vision_api.detect_human(frame_jpg)
        #annotated_jpg = build_visual_output(util.decode_jpg(frame_jpg), preds)

        outputMsg = FrameMsg.new_message()
        #outputMsg.frameData = annotated_jpg
        outputMsg.frameData = frame_jpg
        outputMsg.timestampEpoch = timestamp
        util.write_packet(subscriber_sock, outputMsg.to_bytes())

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
            sys.stderr.write("Waiting for connection on {}\n".format(s))
            conn, addr = s.accept()
            sys.stderr.write("Recv connection from {} {}\n".format(conn, addr))
            return conn

        conn = connect(s)
        conn2 = connect(t)
        handle_robot(conn, conn2)

    sys.stderr.write("Connection closed\n")
