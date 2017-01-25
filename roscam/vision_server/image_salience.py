import time
import sys
import os

import numpy as np
from keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

model = None

def init(filename):
    global model
    if model is None:
        start_time = time.time()
        model = load_model(filename)
        print("Salience network initialized from {} in {:.3f} sec".format(filename, time.time() - start_time))


def run(pixels, resnet_preds):
    import resnet
    if resnet_preds is None:
        resnet.init()
        resnet_preds = resnet.run(pixels)
    inputs = np.expand_dims(resnet_preds, axis=0)
    preds = model.predict(inputs)
    return preds
