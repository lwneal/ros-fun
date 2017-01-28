import time
import sys
import os

import numpy as np
from PIL import Image
from keras.models import load_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_server import resnet
from shared import nlp_api
from shared import vision_api
from shared import util

MAX_OUTPUT_WORDS = 12
VOCABULARY_SIZE = 28519  # TODO: Get this from the NLP server

model = None

def init(filename):
    global model
    start_time = time.time()
    if model is None:
        resnet.init()
        print("Loading image captioning model from {}".format(filename))
        model = load_model(filename)
        print("Image Captioning model initialized in {:.3f} sec".format(time.time() - start_time))


def run_on_jpg_filename(filename):
    jpg = open(filename).read()
    pixels = util.decode_jpg(jpg)
    resnet_preds = resnet.run(pixels)
    return run(resnet_preds)


def run(resnet_preds, img_height, img_width, box):
    x = extract_features_from_preds(resnet_preds, img_height, img_width, box)
    x = np.expand_dims(x, axis=0)
    onehot = model.predict(x)[0]
    return nlp_api.onehot_to_words(onehot)


def extract_features_from_preds(resnet_preds, img_height, img_width, bbox):
    preds_height, preds_width, preds_depth = resnet_preds.shape
    assert preds_depth == 2048

    # Select a single 2048-dim vector from the center of the bbox
    # TODO: Or average over all vectors in the bbox?
    x0, x1, y0, y1 = bbox
    center_x = ((x0 + x1) / 2.0)  * (float(preds_width) / img_width)
    center_y = ((y0 + y1) / 2.0)  * (float(preds_height) / img_height)
    center_x = np.clip(center_x, 0, preds_width-1)
    center_y = np.clip(center_y, 0, preds_height-1)
    local_preds = resnet_preds[int(center_y), int(center_x)]

    # Also use global context: average over the image
    avg_resnet_preds = resnet_preds.mean(axis=0).mean(axis=0)

    context_vector = np.zeros((5,))
    x0, x1, y0, y1 = bbox
    # Left
    context_vector[0] = float(x0) / img_width
    # Top
    context_vector[1] = float(y0) / img_height
    # Right
    context_vector[2] = float(x1) / img_width
    # Bottom
    context_vector[3] = float(y1) / img_height
    # Size
    context_vector[4] = float((x1 - x0) * (y1 - y0)) / (img_width*img_height)

    # Output: 2048 + 2048 + 5 = 4101
    return np.concatenate((local_preds, avg_resnet_preds, context_vector))


def extract_features(jpg_data, img_width, img_height, bbox):
    #img_height, img_width, channels = img.shape
    resnet_preds = vision_api.run_resnet(jpg_data)
    return extract_features_from_preds(resnet_preds, img_height, img_width, bbox)

