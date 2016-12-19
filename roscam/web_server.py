import time
import flask
from base64 import b64encode

app = flask.Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/visual.jpg')
def visual():
    return app.send_static_file('visual.jpg')

@app.route('/stream')
def stream_it():
    def generate():
        while True:
            time.sleep(.1)
            jpg_data = open('static/visual.jpg').read()
            yield 'data:image/jpeg;base64,{}\n\n'.format(b64encode(jpg_data))
    return flask.Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run()
