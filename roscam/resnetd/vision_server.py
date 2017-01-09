import pickle
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
    #print("Decoded input jpg in {:.3f}".format(time.time() - start_time))

    start_time = time.time()
    preds = neural_network.run(pixels)
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
    packet_type, jpg_image = read_packet(conn)
    preds = resnet_jpg(jpg_image)
    #print("Preds have shape {}".format(preds.shape))

    # Round activations to 8-bit values
    preds = preds.astype(np.uint8)

    data = pickle.dumps(preds)
    write_packet(conn, data)
    compressed = data.encode('zlib')
    print("Wrote pickle packet length {} in {:.3f}s (compressed length {})".format(len(data), time.time() - start_time, len(compressed)))
    #print("Produced JPG length {:10d} in {:.3f}s".format(len(output_img), time.time() - start_time))

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
