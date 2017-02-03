import struct
from PIL import Image
import numpy as np
from StringIO import StringIO

import capnp
from frametalk_capnp import FrameMsg


FONT_FILENAME = "/home/nealla/ros-fun/sans-serif.ttf"


def read_packet(conn):
    packet_type_bytes = conn.recv(1)
    assert packet_type_bytes == '\x01'
    packet_type = ord(packet_type_bytes)
    packet_len_bytes = conn.recv(4)
    packet_len = struct.unpack('!l', packet_len_bytes)[0]
    packet_data = ""
    while len(packet_data) < packet_len:
        packet_data_bytes = conn.recv(packet_len - len(packet_data))
        packet_data = packet_data + packet_data_bytes
    return FrameMsg.from_bytes(packet_data).to_dict()


def write_packet(conn, data):
    packet_type = '\x01'
    encoded_len = struct.pack('!l', len(data))
    conn.send(packet_type + encoded_len + data)


def encode_jpg(pixels):
    img = Image.fromarray(pixels).convert('RGB')
    fp = StringIO()
    img.save(fp, format='JPEG')
    return fp.getvalue()


def decode_jpg(jpg):
    return np.array(Image.open(StringIO(jpg)).convert('RGB'))


def rescale(image, shape):
    # TODO: Get rid of imresize
    from scipy.misc import imresize
    return imresize(image, shape).astype(float)


def build_detection_visualization(frame_jpg, preds, caption=None):
    pixels = decode_jpg(frame_jpg)

    preds = rescale(preds, pixels.shape)

    # Darken and set the red channel to the output of the detector
    pixels[:,:,0] = 0.5 * pixels[:,:,0]
    pixels[:,:,1] = 0.5 * pixels[:,:,1]
    pixels[:,:,2] = 0.5 * pixels[:,:,2]
    pixels[:,:,0] = pixels[:,:,0] + 0.5 * preds

    if caption:
        pixels = draw_text(pixels, caption)

    return encode_jpg(pixels)


def draw_text(image_array, text):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    image_array = image_array.astype(np.uint8)
    img = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_FILENAME, 16)
    #draw.rectangle((0, 0, img.width, 16), fill=(0,0,0,128))
    draw.rectangle((0, 0, img.width, 16), fill=(0,0,0,0))
    draw.text((0, 0), text, (255,255,255), font=font)
    return np.array(img)


def draw_box(pixels, box):
    x0, x1, y0, y1 = box
    x1 -= 1
    y1 -= 1
    pixels[y0, x0:x1] = 255
    pixels[y1, x0:x1] = 255
    pixels[y0:y1, x0] = 255
    pixels[y0:y1, x1] = 255


def file_checksum(filename):
    try:
        return subprocess.check_output(['md5sum', model_filename])
    except:
        return '0'
