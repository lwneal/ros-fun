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

        preds, robotCommand = vision_api.detect_human(frame_jpg)

        annotated_jpg = build_detection_visualization(frame_jpg, preds)

        outputMsg = FrameMsg.new_message()
        outputMsg.frameData = annotated_jpg
        outputMsg.timestampEpoch = timestamp
        util.write_packet(subscriber_sock, outputMsg.to_bytes())


def build_detection_visualization(frame_jpg, preds):
    pixels = util.decode_jpg(frame_jpg)

    preds = util.rescale(preds, pixels.shape)

    # Set the red channel to the output of the detector
    pixels[:,:,0] = 0.5 * pixels[:,:,0] + 0.5 * preds
    return util.encode_jpg(pixels)


if __name__ == '__main__':
    # TODO: Async networking
    # This socket receives video from the robot
    sock_robot_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_robot_video.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Port 3389 is open over the engr network
    sock_robot_video.bind(('0.0.0.0', 3389))
    sock_robot_video.listen(1)

    # This socket sends commands to the robot
    sock_robot_control = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_robot_control.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Port 5432 is open over the engr network
    sock_robot_control.bind(('0.0.0.0', 5432))
    sock_robot_control.listen(1)

    # This socket relays video to the viewer
    sock_viewer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock_viewer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock_viewer.bind(('0.0.0.0', 1235))
    sock_viewer.listen(1)

    while True:
        print("Waiting for connection")

        def connect(s):
            conn, addr = s.accept()
            return conn

        print("Waiting for robot video connection")
        conn = connect(sock_robot_video)

        #print("Waiting for robot control socket connection")
        #conn = connect(sock_robot_control)

        print("Waiting for website viewer connection")
        conn2 = connect(sock_viewer)
        handle_robot(conn, conn2)
