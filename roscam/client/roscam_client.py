"""
This program runs on the robot. It requires ROS.
It grabs every frame of video and sends it to roscam.
"""
import sys
import time
import rospy
import docopt
import message_filters
import theora_image_transport.msg
from sensor_msgs.msg import CompressedImage

import network

def video_callback(msg):
    frame_data = msg.data
    timestamp = msg.header.stamp.to_sec()
    latency = time.time() - timestamp
    print("Saving video frame size {:8d} age {:.3f} seconds".format(len(frame_data), latency))
    msg = frame_data
    network.send(frame_data)


def main(topic):
    print("Subscribing to video topic {}".format(topic))
    rospy.init_node('roscam')
    sub_img = message_filters.Subscriber(topic, theora_image_transport.msg.Packet)
    sub_img.registerCallback(video_callback)
    print("Entering ROS spin event loop")
    rospy.spin()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: {} /camera/rgb/image_raw/compressed".format(sys.argv[0]))
        exit()
    main(sys.argv[1])
