"""
Listens on port 1234 for a connection from roscam_client.py, then accepts frames
"""
import socket
import struct
import sys
import os

def handle_robot(robot_sock, subscriber_sock):
    while True:
        packet_type, jpg_data = read_packet(robot_sock)
        output_img = call_vision_server(jpg_data)
        write_packet(subscriber_sock, output_img)

def call_vision_server(jpg_data):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.connect(('127.0.0.1', 1237))
    write_packet(s, jpg_data)
    _, jpg_result = read_packet(s)
    return jpg_result

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

    from threading import Thread
    from multiprocessing.pool import ThreadPool
    pool = ThreadPool(processes=2)

    conn = connect(s)
    conn2 = connect(t)

    if os.fork():
        print("Forked client to handle sockets {}, {}".format(conn, conn2))
    else:
        handle_robot(conn, conn2)

sys.stderr.write("Connection closed\n")
