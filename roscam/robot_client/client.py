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
roslib.load_manifest('pr2_position_scripts')
from theora_image_transport.msg import Packet
from sensor_msgs.msg import CompressedImage
import actionlib
from actionlib_msgs.msg import *
from pr2_controllers_msgs.msg import *
from geometry_msgs.msg import *

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

print("Initializing head control")
client = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
client.wait_for_server()
print("Head control initialized")

g = PointHeadGoal()
g.target.point.x = 10.0
g.target.point.y = 0
g.target.point.z = 0
g.min_duration = rospy.Duration(.5)

# Set head to straight forward
g.target.header.frame_id = '/base_link'
client.send_goal(g)

print("Robot Control client ready...")


last_moved_at = 0
def read_from_socket():
    global last_moved_at
    try:
        while running:
            print("Reading packet...")
            msg = util.read_packet(s)
            command = msg['robotCommand']
            dx = command['headRelAzumith']
            dy = command['headRelAltitude']
            print("Command: move head {} {}".format(dx, dy))
            g.target.header.frame_id = '/wide_stereo_r_stereo_camera_frame'
            g.target.point.y = -.1 * dx
            #g.target.point.y = .05 * dx
            if time.time() - last_moved_at > .0:
                client.send_goal(g)
                last_moved_at = time.time()
    except:
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
