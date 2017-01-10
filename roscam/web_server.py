"""
An HTTP video streamer
"""
import time
import sys
import struct
import flask
import socket
from base64 import b64encode

import util

CLOUD_SERVER = ('localhost', 1235)
app = flask.Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/<path:path>')
def get_static(path):
    return flask.send_from_directory('static', path)


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
                msg = util.read_packet(s)
                jpg_data = msg['frameData']
                yield 'data:image/jpeg;base64,{}\n\n'.format(b64encode(jpg_data))
        except socket.timeout:
            sys.stderr.write('Connection to {} timed out\n'.format(CLOUD_SERVER))
    return flask.Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run('0.0.0.0', 8005)
