"""
This program runs on the robot. It requires ROS.
It grabs every frame of video and sends it to roscam.
"""
import os
import sys
import struct
import time
import socket
import message_filters
import threading
import random

import rospy
import roslib
from theora_image_transport.msg import Packet
from sensor_msgs.msg import CompressedImage

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared import util

import capnp
from frametalk_capnp import FrameMsg

rospy.init_node('roscam_robot_control', anonymous=True)

# Port 3389 is open in the engr network
PORT = 3389
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

MAX_FPS = 10
last_sent_at = 0
running = True


TALK_SECONDS = 10.0
last_talked_at = 0
def read_from_socket():
    global last_excited_at
    global last_talked_at
    try:
        while running:
            msg = util.read_packet(s)
            command = msg['robotCommand']
            dy = command['headRelAltitude']
            if time.time() - last_talked_at > TALK_SECONDS:
                print("Talk: {}".format(command['descriptiveStatement']))
                last_talked_at = time.time()
                # TODO: security
                os.system('echo "{}" | festival --tts'.format(command['descriptiveStatement']))
    except Exception as e:
        print e
        exit()


def main(topic, server_ip):
    print("Connecting to server")
    s.connect((server_ip, PORT))

    print("Subscribing to video topic {}".format(topic))
    sub_img = message_filters.Subscriber(topic, CompressedImage)
    sub_img.registerCallback(video_callback)
    print("Entering ROS spin event loop")
    thread_recv = threading.Thread()
    thread_recv.run = read_from_socket
    thread_recv.start()
    rospy.spin()


def video_callback(ros_msg):
    global last_sent_at
    msg = FrameMsg.new_message()
    msg.timestampEpoch = ros_msg.header.stamp.to_sec()
    msg.frameData = ros_msg.data

    if time.time() - last_sent_at > 1.0 / MAX_FPS:
        send_frame(msg)
        last_sent_at = time.time()
    else:
        pass  # Skip this frame


def send_frame(out_msg):
    global running
    latency = time.time() - out_msg.timestampEpoch
    #print('Streaming video frame size {:8d} age {:.3f}s'.format(len(out_msg.frameData), latency))
    try:
        util.write_packet(s, out_msg.to_bytes())
    except:
        print("Cloud server connection closed, exiting now")
        running = False
        rospy.signal_shutdown('socket dead')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: {} /camera/rgb/image_raw/compressed 127.1.2.3".format(sys.argv[0]))
        exit()
    server_ip = sys.argv[2]
    print("ROS-connected client uploading to server at {}".format(server_ip))
    main(sys.argv[1], sys.argv[2])
