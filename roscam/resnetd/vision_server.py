import time
import sys
import struct
import os
import socket
from cStringIO import StringIO
import numpy as np
from PIL import Image
from scipy.misc import imresize

import neural_network


neural_network.init()


def resnet_jpg(jpg_data):
    pixels = decode_jpg(jpg_data)
    preds = neural_network.run(pixels)
    return encode_jpg(preds.reshape((2048, -1)))


def decode_jpg(jpg_data):
    fp = StringIO(jpg_data)
    img = Image.open(fp)
    return np.array(img.convert('RGB'))


def encode_jpg(pixels):
    pil_img = Image.fromarray(pixels.astype(np.uint8))
    fp = StringIO()
    pil_img.save(fp, format='JPEG')
    return fp.getvalue()


def read_packet(conn):
    packet_type_bytes = conn.recv(1)
    packet_type = ord(packet_type_bytes)

    packet_len_bytes = conn.recv(4)
    packet_len = struct.unpack('!l', packet_len_bytes)[0]
    sys.stderr.write("Got packet type {} length {}... ".format(packet_type, packet_len))

    packet_data = ""
    while len(packet_data) < packet_len:
        packet_data_bytes = conn.recv(packet_len - len(packet_data))
        packet_data = packet_data + packet_data_bytes
    return packet_type, packet_data


def write_packet(conn, data):
    packet_type = '\x42'
    encoded_len = struct.pack('!l', len(data))
    conn.send(packet_type + encoded_len + data)


def handle_client(conn):
    start_time = time.time()
    packet_type, jpg_image = read_packet(conn)
    print("running resnet")
    new_image = resnet_jpg(jpg_image)
    print("write_packet")
    write_packet(conn, new_image)
    print("finished handle_client")
    print("handle_client in {:.2f}".format(time.time() - start_time))

if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 1237))
    s.listen(1)

    while True:
        print("socket accept()")
        conn, addr = s.accept()
        print("handle_client")
        handle_client(conn)
