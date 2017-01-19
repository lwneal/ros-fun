from StringIO import StringIO
import json
import pickle
import flask
import numpy as np

import word_vector


app = flask.Flask(__name__)

# TODO: API to get vocabulary length

@app.route('/words_to_vec', methods=['GET', 'POST'])
def words_to_vec():
    text = flask.request.values.get('text')
    dimensionality = flask.request.values.get('dimensionality') or 50
    pad_to_length = flask.request.values.get('pad_to_length') or 16

    # Tuple of (word_vectors, one-hot)
    vectors = word_vector.vectorize(text)
    indices = word_vector.text_to_idx(text)

    fp = StringIO()
    pickle.dump(zip(vectors, indices), fp)
    buff = fp.getvalue()
    print("Returning word2vec output length {}".format(len(buff)))
    return flask.Response(buff, mimetype='application/octet-stream')


@app.route('/indices_to_words', methods=['GET', 'POST'])
def indices_to_words():
    text = flask.request.values.get('indices')
    indices = json.loads(text)
    words = [word_vector.word_list[i] for i in indices]
    return flask.Response(' '.join(words), mimetype='text/plain')
    

if __name__ == '__main__':
    word_vector.init()
    app.run(port=8010)
