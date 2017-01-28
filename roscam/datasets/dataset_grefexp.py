import os
import sys
import json

import redis
import numpy as np
import dataset_coco

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util

DATA_DIR = '/home/nealla/data'

KEY_GREFEXP_TRAIN = 'dataset_grefexp_train'
KEY_GREFEXP_VAL = 'dataset_grefexp_val'

conn = redis.Redis()


def random_annotation(reference_key=KEY_GREFEXP_TRAIN):
    key = conn.srandmember(reference_key)
    return get_annotation_for_key(key)


def get_all_keys(reference_key=KEY_GREFEXP_VAL):
    keys = conn.smembers(reference_key)
    return keys


def get_annotation_for_key(key):
    grefexp = json.loads(conn.get(key))
    anno_key = 'coco2014_anno_{}'.format(grefexp['annotation_id'])
    anno = json.loads(conn.get(anno_key))
    img_key = 'coco2014_img_{}'.format(anno['image_id'])
    img_meta = json.loads(conn.get(img_key))

    jpg_data = open(os.path.join(DATA_DIR, img_meta['filename'])).read()
    return grefexp, anno, img_meta, jpg_data
