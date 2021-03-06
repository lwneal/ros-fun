import os
import time
import sys
import struct
import flask
import socket
from base64 import b64encode

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import util
from shared import vision_api
from cloud_server import block_storage

CLOUD_SERVER = ('localhost', 1235)
app = flask.Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/watch')
def watch_endpoint():
    return app.send_static_file('watch.html')


@app.route('/<path:path>')
def get_static(path):
    return flask.send_from_directory('static', path)


@app.route('/stream')
def stream_live():
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


@app.route('/stream_recorded')
def stream_recorded():
    def generate():
        # TODO: Session selection, for now choose a random one
        import random
        session = random.choice(block_storage.get_sessions())
        for ts, jpg_data in block_storage.read_frames(session['id']):
            time.sleep(.05)  # TODO: rate limit?
            preds, info = vision_api.detect_human(jpg_data)
            caption = info['descriptiveStatement']
            print caption
            annotated_jpg = util.build_detection_visualization(jpg_data, preds, caption)
            yield 'data:image/jpeg;base64,{}\n\n'.format(b64encode(annotated_jpg))
    return flask.Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Ports 5800 and 5900 are open on the ENGR network
    app.run('0.0.0.0', 5800)
