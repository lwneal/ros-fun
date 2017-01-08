import os
import json
import urllib

import redis
import numpy as np
from PIL import Image
from skimage.draw import polygon

COCO_DIR = '/home/nealla/data/train2014'
REFERENCE_KEY = 'dataset_vg_keys'


conn = redis.Redis()
key_count = conn.scard(REFERENCE_KEY)
print("COCO dataset client connected to {}, reading {} keys".format(conn, key_count))


def random_image():
    image_key = conn.srandmember(REFERENCE_KEY)
    img_meta = json.loads(conn.get(image_key))
    pixels = load_jpg(img_meta['file_name'], img_meta['coco_url'])
    assert img_meta['height'] == pixels.shape[0]
    return img_meta, pixels


def load_jpg(image_name, url=None):
    filename = os.path.join(COCO_DIR, image_name)
    if not os.path.exists(filename) and url:
        print("Image {} not cached, downloading from {}".format(image_name, url))
        urllib.urlretrieve(url, filename)
    img = Image.open(filename)
    return np.array(img)


def human_detection_mask(img_meta, pixels):
    CATEGORY_ID_PERSON = 1  # see coco_categories.json
    annotations = [a for a in img_meta['annotations'] if a['category_id'] == CATEGORY_ID_PERSON]
    return build_mask(img_meta['width'], img_meta['height'], annotations)


def build_mask(width, height, annotations):
    mask = np.zeros((height, width), dtype=np.float32)
    for anno in annotations:
        for poly in anno['segmentation']:
            x_coords = []
            y_coords = []
            i = 0
            while i < len(poly):
                x_coords.append(poly[i])
                y_coords.append(poly[i + 1])
                i += 2
            # Some polygons are malformed/out of bounds
            try:
                rr, cc = polygon(y_coords, x_coords)
                mask[rr, cc] = 1.0
            except:
                print("Error: Mask skipping invalid annotation {}".format(chosen_annotation))
    return mask

img_meta, pixels = random_image()
mask = human_detection_mask(img_meta, pixels)
print("Test: got random image shape {} and binary mask for human detection shape {}".format(pixels.shape, mask.shape))
