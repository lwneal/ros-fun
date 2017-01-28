#!/usr/bin/env python
import random
import os
import sys
import math

from keras.models import load_model
import numpy as np
from PIL import Image
import nltk

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from shared import nlp_api
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net


def get_grefexp(key):
    grefexp, anno, img_meta, jpg_data = dataset_grefexp.get_annotation_for_key(key)
    x0, width, y0, height = anno['bbox']
    box = (x0, x0 + width, y0, y0 + height)
    texts = [refexp['raw'] for refexp in grefexp['refexps']]
    return jpg_data, img_meta['width'], img_meta['height'], box, texts


def get_validation_set():
    keys = dataset_grefexp.get_all_keys()
    print("Loaded {} validation examples".format(len(keys)))
    for key in keys:
        jpg_data, width, height, box, texts = get_grefexp(key)
        x = image_caption.extract_features(jpg_data, width, height, box)
        x = np.expand_dims(x, axis=0)
        #print("Training on word sequence: {}".format(nlp_api.onehot_to_words(y)))
        yield x, texts


def evaluate(model):
    bleu_count = 0
    bleu1_sum = 0
    bleu2_sum = 0
    bleu4_sum = 0
    import time
    now = time.time()
    runtimes = []
    for x, correct_texts in get_validation_set():
        preds = model.predict(x)
        output_words = nlp_api.onehot_to_words(preds.reshape(preds.shape[1:]))
        output_words = output_words.replace('001', '')  # Remove end token
        bleu1_sum += bleu(output_words, correct_texts, n=1)
        bleu2_sum += bleu(output_words, correct_texts, n=2)
        bleu4_sum += bleu(output_words, correct_texts, n=4)
        bleu_count += 1
        print("Examples:\t{}\tAvg BLEU-1 score:\t{:.3f}\tBLEU-2:\t{:.3f}\tBLEU-4:{:.3f}".format(
            bleu_count, bleu1_sum / bleu_count, bleu2_sum / bleu_count,
            bleu4_sum / bleu_count))
        runtimes.append(time.time() - now)
        print("Average runtime {:.3f}".format(sum(runtimes) / len(runtimes)))
        now = time.time()


def bleu(candidate, references, n):
    ground_truths = [r.lower().split() for r in references]
    generated = candidate.lower().split()
    if n == 1:
        weights = (1.0,)
    elif n == 2:
        weights = (.5, .5)
    elif n == 4:
        weights = (.25, .25, .25, .25)
    return nltk.translate.bleu_score.sentence_bleu(ground_truths, generated, weights)


if __name__ == '__main__':
    input_filename = sys.argv[1]
    if len(sys.argv) > 2:
        input_filename = sys.argv[2]

    model = load_model(input_filename)
    evaluate(model)
