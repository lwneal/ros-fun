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
    visual_activations = resnet(jpg_data)
    visual_activations = np.expand_dims(visual_activations, axis=0)
    preds = model.predict(visual_activations)
    return preds.reshape(preds.shape[1:-1])
