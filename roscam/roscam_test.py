"""
This program runs on the robot. It requires ROS.
It grabs every frame of video and sends it to roscam.
"""
import os
import sys
import time
import rospy
import docopt
import message_filters
import theora_image_transport.msg
from sensor_msgs.msg import CompressedImage


video_file = 'video.mjpeg'
timestamp_file = 'timestamps.txt'


def store(frame, timestamp):
    timestamp_str = '{:.03f}\n'.format(timestamp)
    open(video_file, 'ab').write(frame)
    open(timestamp_file, 'a').write(timestamp_str)


def check():
    video_data = open(video_file).read()
    timestamps = open(timestamp_file).readlines()
    return len(video_data), len(timestamps)


def video_callback(msg):
    frame_data = msg.data
    timestamp = msg.header.stamp.to_sec()
    latency = time.time() - timestamp
    print("Saving video frame size {:8d} age {:.3f} seconds".format(len(frame_data), latency))
    store(frame_data, timestamp)


def main(topic):
    print("Subscribing to video topic {}".format(topic))
    rospy.init_node('roscam')
    sub_img = message_filters.Subscriber(topic, CompressedImage)
    sub_img.registerCallback(video_callback)
    print("Entering ROS spin event loop")
    rospy.spin()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} /camera/rgb/image_raw/compressed".format(sys.argv[0]))
        exit()
    main(sys.argv[1])



