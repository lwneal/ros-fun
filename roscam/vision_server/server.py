import pickle
import time
import sys
import struct
import os
import socket
from cStringIO import StringIO

import numpy as np
from PIL import Image
import capnp

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import resnet
import human_detector
from shared import util
from frametalk_capnp import FrameMsg
import robot_command
import image_caption


def resnet_request(pixels):
    preds = resnet.run(pixels)
    # Remove extra dimension
    preds = preds.reshape(preds.shape[1:])
    # Round activations to 8-bit values
    preds = preds.astype(np.uint8)
    # Output Shape: (15, 20, 2048)
    return preds


# Note: Also returns a robotCommand
def detect_human_request(pixels):
    resnet_preds = resnet.run(pixels)

    # Detect humans to point head
    preds = human_detector.run(pixels, resnet_preds=resnet_preds)

    # Remove extra dimensions
    preds = preds.reshape(preds.shape[1:-1]) * 255
    # Round activations to 8-bit values
    preds = preds.astype(np.uint8)

    # Generate text for TTS
    # TODO: Which bounding box should be used?
    img_height, img_width, channels = pixels.shape
    bbox = (0, img_width, 0, img_height)
    words = image_caption.run(resnet_preds, img_height, img_width, bbox)
    print words.replace('001', '')

    # Output Shape: (15, 20)
    robotCommand = robot_command.point_head_toward_human(preds)
    robotCommand.descriptiveStatement = words.replace('001','')
    return preds, robotCommand


def handle_client(conn):
    start_time = time.time()
    msg = util.read_packet(conn)
    jpg_image = msg['frameData']
    requestType = msg['visionType']

    pixels = util.decode_jpg(jpg_image)

    #print "Got request len {} with type {}".format(len(jpg_image), requestType)
    if requestType == 'resNet50':
        # Return rounded preds as pickle
        preds = resnet_request(pixels)
        robotCommand = None
    elif requestType == 'detectHuman':
        # Detect humans, return pickled preds
        preds, robotCommand = detect_human_request(pixels)

    outputMsg = FrameMsg.new_message()
    outputMsg.frameData = pickle.dumps(preds)
    if robotCommand:
        #print("Sending robot command: {}".format(robotCommand.to_dict()))
        outputMsg.robotCommand = robotCommand
    util.write_packet(conn, outputMsg.to_bytes())
    print("Finished request type {} in {:.3f}s".format(requestType, time.time() - start_time))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: server.py saliency_model.h5 image_caption_model.h5")
        exit()

    human_detector.init(filename=sys.argv[1])
    image_caption.init(filename=sys.argv[2])

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('0.0.0.0', 1237))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        start_time = time.time()
        handle_client(conn)
