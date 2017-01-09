import struct
from PIL import Image
import numpy as np
from StringIO import StringIO


def read_packet(conn):
    packet_type_bytes = conn.recv(1)
    packet_type = ord(packet_type_bytes)
    packet_len_bytes = conn.recv(4)
    packet_len = struct.unpack('!l', packet_len_bytes)[0]
    packet_data = ""
    while len(packet_data) < packet_len:
        packet_data_bytes = conn.recv(packet_len - len(packet_data))
        packet_data = packet_data + packet_data_bytes
    return packet_type, packet_data


def write_packet(conn, data):
    packet_type = '\x42'
    encoded_len = struct.pack('!l', len(data))
    conn.send(packet_type + encoded_len + data)


def encode_jpg(pixels):
    img = Image.fromarray(pixels).convert('RGB')
    fp = StringIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


def decode_jpg(jpg):
    return np.array(Image.open(StringIO(jpg)))
