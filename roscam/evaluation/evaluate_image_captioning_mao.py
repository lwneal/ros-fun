#!/usr/bin/env python
"""
Usage:
        evaluate --model MODEL.h5 [--count COUNT]

Options:
        --model MODEL.h5    Path to saved HDF5 model compatible with keras.load_model
        --count COUNT       Only evaluate using a subset of COUNT examples
"""
import docopt
import re
import random
import os
import sys
import math
import time

from keras.models import load_model
import numpy as np
from PIL import Image
import rouge_scorer
import bleu_scorer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util
from shared import nlp_api
from datasets import dataset_grefexp
from interfaces import image_caption
from networks import mao_net


def get_grefexp(key):
    grefexp, anno, img_meta, jpg_data = dataset_grefexp.get_annotation_for_key(key)
    x0, y0, width, height = anno['bbox']
    box = (x0, x0 + width, y0, y0 + height)
    texts = [refexp['raw'] for refexp in grefexp['refexps']]
    return jpg_data, img_meta['width'], img_meta['height'], box, texts


def get_validation_set(count=None):
    keys = dataset_grefexp.get_all_keys()
    if count:
        keys = list(keys)
        random.shuffle(keys)
        keys = keys[:count]
    print("Loaded {} validation examples".format(len(keys)))
    for key in keys:
        jpg_data, width, height, box, texts = get_grefexp(key)
        from vision_server import resnet
        resnet.init()
        pixels = util.decode_jpg(jpg_data)
        resnet_preds = resnet.run(pixels)
        x = image_caption.extract_features_from_preds(resnet_preds, width, height, box)
        x = np.expand_dims(x, axis=0)
        yield x, texts


def evaluate(model, **kwargs):
    score_names = ['BLEU1', 'BLEU2', 'ROUGE']
    scores = compute_scores(model, **kwargs)
    for name, score_list in zip(score_names, scores):
        print name
        from scipy import stats
        print stats.describe(score_list)


def compute_scores(model, count=None):
    score_list = []
    for x, reference_texts in get_validation_set(count=count):
        indices = mao_net.predict(model, x, [nlp_api.START_TOKEN_IDX])

        candidate = strip(nlp_api.indices_to_words(indices))
        references = [strip(r) for r in reference_texts]

        bleu1_score, bleu2_score = bleu(candidate, references)
        rouge_score = rouge([candidate], reference_texts)

        print('{:.3f} {:.3f} {:.3f} {} ({})'.format(bleu1_score, bleu2_score, rouge_score, candidate, references[0]))
        score_list.append((bleu1_score, bleu2_score, rouge_score))
    return zip(*score_list)


def strip(text):
    # Remove the START_TOKEN
    text = text.replace('000', '')
    # Remove all text after the first END_TOKEN
    end_idx = text.find('001')
    if end_idx >= 0:
        text = text[:end_idx]
    # Remove non-alphanumeric characters and lowercase everything
    return re.sub(r'\W+', ' ', text.lower()).strip()


def bleu(candidate, references):
    scores, _ = bleu_scorer.BleuScorer(candidate, references, n=2).compute_score(option='closest')
    return scores


def rouge(candidate, references):
    return rouge_scorer.Rouge().calc_score(candidate, references)


if __name__ == '__main__':
    opt = docopt.docopt(__doc__)
    model_filename = opt['--model']
    count = opt['--count']
    if count:
        count = int(count)

    model = load_model(model_filename)
    evaluate(model, count=count)
