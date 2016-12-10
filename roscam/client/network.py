"""
This code sends stuff over the Internet.
"""
import socket

SERVER = 'localhost'
PORT = 33133

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((SERVER, PORT))

def send(msg):
    s.send(msg)
