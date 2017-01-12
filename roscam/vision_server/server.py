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
from frametalk_capnp import FrameMsg


resnet.init()


def resnet_jpg(jpg_data):
    start_time = time.time()
    pixels = decode_jpg(jpg_data)
    #print("Decoded input jpg in {:.3f}".format(time.time() - start_time))

    start_time = time.time()
    preds = resnet.run(pixels)
    #print("Ran ResNet50 in {:.3f}s".format(time.time() - start_time))
    # Remove extra dimension
    return preds.reshape(preds.shape[1:])


def decode_jpg(jpg_data):
    fp = StringIO(jpg_data)
    img = Image.open(fp)
    return np.array(img.convert('RGB'))


def encode_jpg(pixels):
    pil_img = Image.fromarray(pixels.astype(np.uint8))
    fp = StringIO()
    pil_img.save(fp, format='JPEG', quality=100)
    return fp.getvalue()


def handle_client(conn):
    start_time = time.time()
    msg = util.read_packet(conn)
    jpg_image = msg['frameData']
    preds = resnet_jpg(jpg_image)

    # Round activations to 8-bit values
    preds = preds.astype(np.uint8)

    outputMsg = FrameMsg.new_message()
    outputMsg.frameData = pickle.dumps(preds)
    util.write_packet(conn, outputMsg.to_bytes())

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 1237))
    s.listen(1)

    while True:
        #print("Waiting for connection...")
        conn, addr = s.accept()
        start_time = time.time()
        handle_client(conn)
