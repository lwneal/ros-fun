import struct
from PIL import Image
import numpy as np
from StringIO import StringIO

import capnp
from frametalk_capnp import FrameMsg


def read_packet(conn):
    packet_type_bytes = conn.recv(1)
    assert packet_type_bytes == '\x01'
    packet_type = ord(packet_type_bytes)
    packet_len_bytes = conn.recv(4)
    packet_len = struct.unpack('!l', packet_len_bytes)[0]
    packet_data = ""
    while len(packet_data) < packet_len:
        packet_data_bytes = conn.recv(packet_len - len(packet_data))
        packet_data = packet_data + packet_data_bytes
    return FrameMsg.from_bytes(packet_data).to_dict()


def write_packet(conn, data):
    packet_type = '\x01'
    encoded_len = struct.pack('!l', len(data))
    conn.send(packet_type + encoded_len + data)


def encode_jpg(pixels):
    img = Image.fromarray(pixels).convert('RGB')
    fp = StringIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


def decode_jpg(jpg):
    return np.array(Image.open(StringIO(jpg)).convert('RGB'))


def rescale(image, shape):
    # TODO: Get rid of imresize
    from scipy.misc import imresize
    return imresize(image, shape).astype(float)

