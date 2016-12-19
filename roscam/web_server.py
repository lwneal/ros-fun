import time
import sys
import struct
import flask
import socket
from base64 import b64encode

from vision import computer_vision

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
        sys.stderr.write("connecting to localhost ...\n")
        s.connect(('localhost', 1235))
        sys.stderr.write("socket connected\n")
        while True:
            jpg_data = read_packet_from_socket(s)
            processed_jpg = computer_vision(jpg_data)
            yield 'data:image/jpeg;base64,{}\n\n'.format(b64encode(processed_jpg))
    return flask.Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run('0.0.0.0', 8005)
