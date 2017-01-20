import time
import sys
import os

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import nlp_api
from shared import util

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

def run(resnet_preds):
    avg_pred = resnet_preds.mean(axis=0).mean(axis=0)
    x = np.expand_dims(avg_pred, axis=0)
    preds = model.predict(x)
    return nlp_api.onehot_to_words(preds[0])
