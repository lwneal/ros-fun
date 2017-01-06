"""
Listens on port 1234 for a connection from roscam_client.py, then accepts frames
"""
import socket
import struct
import sys
import os
import util

def handle_robot(robot_sock, subscriber_sock):
    while True:
        packet_type, jpg_data = util.read_packet(robot_sock)
        output_img = call_vision_server(jpg_data)
        util.write_packet(subscriber_sock, output_img)

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
