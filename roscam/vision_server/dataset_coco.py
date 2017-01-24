import sys
import os
import json
import urllib

import redis
import numpy as np
from PIL import Image
from skimage.draw import polygon

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared import util

DATA_DIR = '/home/nealla/data'

KEY_COCO2014_IMAGES_TRAIN = 'dataset_coco2014_images_train'
KEY_COCO2014_IMAGES_VAL = 'dataset_coco2014_images_val'
KEY_COCO2014_ANNOTATIONS_TRAIN = 'dataset_coco2014_annotations_train'
KEY_COCO2014_ANNOTATIONS_VAL = 'dataset_coco2014_annotations_val'


conn = redis.Redis()


def random_image(reference_key=KEY_COCO2014_IMAGES_TRAIN):
    image_key = conn.srandmember(reference_key)
    img_meta = json.loads(conn.get(image_key))
    filename = os.path.join(DATA_DIR, img_meta['filename'])
    pixels = util.decode_jpg(open(filename).read())
    assert img_meta['height'] == pixels.shape[0]
    return img_meta, pixels


def human_detection_mask(img_meta):
    CATEGORY_ID_PERSON = 1  # see coco_categories.json
    annotations = img_meta.get('annotations', [])
    annotations = [get_annotation(anno_id) for anno_id in annotations]
    annotations = [anno for anno in annotations if anno['category_id'] == CATEGORY_ID_PERSON]
    return build_mask(img_meta['width'], img_meta['height'], annotations)


def get_annotation(annotation_id):
    return json.loads(conn.get('coco2014_anno_{}'.format(annotation_id)))


def build_mask(width, height, annotations):
    mask = np.zeros((height, width), dtype=np.float32)
    for anno in annotations:
        seg = anno['segmentation']
        if type(seg) is list:
            # Normal mask: list of polygons
            for poly in seg:
                add_poly_to_mask(poly, mask)
        else:
            # This is a kooky RLE mask, decode it
            decoded = decodeMask(seg)
            mask[np.where(decoded > 0)] = 1.0
    return mask


def add_poly_to_mask(poly, mask):
    x_coords = []
    y_coords = []
    i = 0
    while i < len(poly):
        x_coords.append(poly[i])
        y_coords.append(poly[i + 1])
        i += 2
    rr, cc = polygon(y_coords, x_coords)
    mask[rr, cc] = 1.0


def decodeMask(R):
    """
    Decode binary mask M encoded via run-length encoding.
    :param   R (object RLE)    : run-length encoding of binary mask
    :return: M (bool 2D array) : decoded binary mask
    """
    N = len(R['counts'])
    M = np.zeros( (R['size'][0]*R['size'][1], ))
    n = 0
    val = 1
    for pos in range(N):
        val = not val
        for c in range(R['counts'][pos]):
            R['counts'][pos]
            M[n] = val
            n += 1
    return M.reshape((R['size']), order='F')
