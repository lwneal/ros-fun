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
    grefexp_key = conn.srandmember(reference_key)
    grefexp = json.loads(conn.get(grefexp_key))
    anno_key = 'coco2014_anno_{}'.format(grefexp['annotation_id'])
    anno = json.loads(conn.get(anno_key))
    img_key = 'coco2014_img_{}'.format(anno['image_id'])
    img_meta = json.loads(conn.get(img_key))

    jpg_data = open(os.path.join(DATA_DIR, img_meta['filename'])).read()
    pixels = util.decode_jpg(jpg_data)
    return grefexp, anno, img_meta, pixels
