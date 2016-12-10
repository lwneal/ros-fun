import sys
import time
import rospy
import docopt
import message_filters
import theora_image_transport.msg

import block_storage


def video_callback(msg):
    frame_data = msg.data
    timestamp = msg.header.stamp.to_sec()
    latency = time.time() - timestamp
    print("Saving video frame size {:8d} age {:.3f} seconds".format(len(frame_data), latency))
    block_storage.store(frame_data, timestamp)


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
