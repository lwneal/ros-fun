import time
import sys
import struct
import os
import socket
from cStringIO import StringIO
import numpy as np
from PIL import Image

sys.path.append('..')
import neural_network
from util import read_packet, write_packet


neural_network.init()


def resnet_jpg(jpg_data):
    start_time = time.time()
    pixels = decode_jpg(jpg_data)
    print("Decoded jpg in {:.2f}".format(time.time() - start_time))

    start_time = time.time()
    preds = neural_network.run(pixels)
    print("Ran ResNet50 in {:.2f}s".format(time.time() - start_time))
    return preds


def decode_jpg(jpg_data):
    fp = StringIO(jpg_data)
    img = Image.open(fp)
    return np.array(img.convert('RGB'))


def encode_jpg(pixels):
    pil_img = Image.fromarray(pixels.astype(np.uint8))
    fp = StringIO()
    pil_img.save(fp, format='JPEG')
    return fp.getvalue()


def handle_client(conn):
    start_time = time.time()
    packet_type, jpg_image = read_packet(conn)
    preds = resnet_jpg(jpg_image)
    print("Preds have shape {}".format(preds.shape))

    output_img = encode_jpg(preds.reshape((2048, -1)))
    print("Processed image output size {} in {:.2f}".format(len(output_img), time.time() - start_time))

    start_time = time.time()
    # Data: Height, width, encoded JPG
    height = struct.pack('!l', preds.shape[1])
    width = struct.pack('!l', preds.shape[2])
    write_packet(conn, height + width + output_img)
    print("Wrote output in {:.2f}".format(time.time() - start_time))

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 1237))
    s.listen(1)

    while True:
        print("Waiting for connection")
        conn, addr = s.accept()
        start_time = time.time()
        handle_client(conn)
        print("Handled client in {:.2f}".format(time.time() - start_time))
