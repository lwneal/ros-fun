import time
import sys
import os

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import nlp_api
from shared import util
import mao_net as network

model = None

def init(filename='/home/nealla/models/image_captioning_lstm_silly.h5'):
    global model
    start_time = time.time()
    if model is None:
        import resnet
        resnet.init()
        from keras.models import load_model
        print("Loading image captioning model from {}".format(filename))
        model = load_model(filename)
        print("Image Captioning model initialized in {:.3f} sec".format(time.time() - start_time))


def run_on_jpg_filename(filename='/tmp/human.jpg'):
    jpg = open(filename).read()
    pixels = util.decode_jpg(jpg)
    import resnet
    resnet_preds = resnet.run(pixels)
    return run(resnet_preds)


def run(resnet_preds, img_height, img_width, box):
    x = network.extract_features_from_preds(resnet_preds, img_height, img_width, box)
    x = np.expand_dims(x, axis=0)
    onehot = model.predict(x)[0]
    return nlp_api.onehot_to_words(onehot)
