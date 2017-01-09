import sys
import os

import numpy as np
from PIL import Image

from train_detector import load_model
from vision_client import resnet


model = None


def init(model_filename):
    global model
    model = load_model(model_filename)


def detect_human(jpg_data):
    print("detect_human visual_activations resnet")
    visual_activations = resnet(jpg_data)
    print("detect_human expand_dims")
    visual_activations = np.expand_dims(visual_activations, axis=0)
    print("detect_human predict")
    preds = model.predict(visual_activations)
    print("detect_human reshape")
    return preds.reshape(preds.shape[1:-1])
