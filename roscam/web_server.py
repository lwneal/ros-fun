"""
An HTTP video streamer
"""
import time
import sys
import struct
import flask
import socket
from base64 import b64encode

CLOUD_SERVER = ('localhost', 1235)
app = flask.Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_static(path):
    return flask.send_from_directory('static', path)


def read_packet_from_socket(sock):
    packet_type = ord(sock.recv(1))
    packet_length = struct.unpack('!l', sock.recv(4))[0]
    packet_data = []
    bytes_read = 0
    while bytes_read < packet_length:
        new_data = sock.recv(packet_length - bytes_read)
        bytes_read += len(new_data)
        packet_data.append(new_data)
    return ''.join(packet_data)


@app.route('/stream')
def stream_it():
    def generate():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3.0)
        sys.stderr.write("connecting to {} ...\n".format(CLOUD_SERVER))
        s.connect(CLOUD_SERVER)
        sys.stderr.write("socket connected\n")
        try:
            while True:
                jpg_data = read_packet_from_socket(s)
                yield 'data:image/jpeg;base64,{}\n\n'.format(b64encode(jpg_data))
        except socket.timeout:
            sys.stderr.write('Connection to {} timed out\n'.format(CLOUD_SERVER))
    return flask.Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run('0.0.0.0', 8005)
