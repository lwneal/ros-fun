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
from shared.nlp_api import VOCABULARY_SIZE, END_TOKEN_IDX
from datasets import dataset_grefexp

model = None

def init(filename):
    global model
    start_time = time.time()
    if model is None:
        print("Loading image captioning model from {}".format(filename))
        model = load_model(filename)
        print("Image Captioning model initialized in {:.3f} sec".format(time.time() - start_time))


def run(resnet_preds, img_height, img_width, box):
    x = extract_features_from_preds(resnet_preds, img_height, img_width, box)
    x = np.expand_dims(x, axis=0)
    onehot = model.predict(x)[0]
    return nlp_api.onehot_to_words(onehot)


def extract_features_from_preds(resnet_preds, img_height, img_width, bbox, pad_to_length=None, average_box=True):
    preds_height, preds_width, preds_depth = resnet_preds.shape
    assert preds_depth == 2048

    # Select a single 2048-dim vector from the center of the bbox
    # Or, average over all vectors in the bbox
    x0, x1, y0, y1 = bbox

    if average_box:
        sx = float(preds_width) / img_width
        sy = float(preds_height) / img_height
        py0, py1 = int(y0*sy), int(y1*sy)
        px0, px1 = int(x0*sx), int(x1*sx)
        if py1 <= py0:
            print("Clipping vertically empty bounding box {}".format(bbox))
            py1 = py0 + 1
        if px1 <= px0:
            print("Clipping horizontally empty bounding box {}".format(bbox))
            px1 = px0 + 1
        local_preds = resnet_preds[py0:py1, px0:px1].mean(axis=0).mean(axis=0)
    else:
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
    visual_input = np.concatenate((local_preds, avg_resnet_preds, context_vector))
    # Output with padding: 12 x 4101
    if pad_to_length:
        padding = np.zeros((pad_to_length,4101))
        padding[0] = visual_input
        return padding
    return visual_input


def predict(model, x_img, x_word, timesteps=10):
    X_img = np.expand_dims(x_img, axis=0)
    X_word = np.expand_dims(x_word, axis=0)
    for _ in range(timesteps):
        preds = model.predict([X_img, X_word])
        next_word = np.argmax(preds, axis=1)
        X_word = np.concatenate([X_word, [next_word]], axis=1)
        X_img = util.extend(X_img)
        if next_word[0] == END_TOKEN_IDX:
            break
    return X_word[0]


def example_mao():
    jpg_data, box, text = dataset_grefexp.random_generation_example()
    pixels = util.decode_jpg(jpg_data)
    preds = resnet.run(pixels)
    width, height, _ = pixels.shape
    x_img = extract_features_from_preds(preds, width, height, box)
    x_img /= (x_img.max() + .1)

    # Train on one word in the sentence
    _, indices = nlp_api.words_to_vec(text)
    if len(indices) < 3:
        print("Warning: invalid caption {}".format(text))
        indices = nlp_api.words_to_vec('nothing')
    return x_img, indices
