"""
Listens on port 1234 for a connection from roscam_client.py, then accepts frames
"""
import socket
import struct
import sys
import os
import util

from PIL import Image
import numpy as np

import human_detector
human_detector.init('human_detector/model.h5')


def build_visual_output(pixels, preds):
    print 'got pixels shape {} preds shape {}'.format(pixels.shape, preds.shape)
    shape = (pixels.shape[1], pixels.shape[0])
    mask = np.array(Image.fromarray(preds * 255).resize(shape)).astype(np.uint8)

    # Red overlay: output of network
    pixels[:,:,0] = mask
    return util.encode_jpg(pixels)
    

def handle_robot(robot_sock, subscriber_sock):
    while True:
        packet_type, frame_jpg = util.read_packet(robot_sock)
        preds = human_detector.detect_human(frame_jpg)
        annotated_jpg = build_visual_output(util.decode_jpg(frame_jpg), preds)
        util.write_packet(subscriber_sock, annotated_jpg)


def call_vision_server(jpg_data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(('127.0.0.1', 1237))
    util.write_packet(s, jpg_data)
    _, jpg_result = util.read_packet(s)
    return jpg_result


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
