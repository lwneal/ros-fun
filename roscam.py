import time
import rospy
import message_filters
import theora_image_transport.msg


def video_callback(msg):
    frame = msg.data
    latency = time.time() - msg.header.stamp.to_sec()
    print("Video frame size {:8d} age {:.2f} seconds".format(len(frame), latency))


def main():
    # Subscribe to the video feed (JPG stream)
    rospy.init_node('roscam')
    topic = '/wide_stereo/left/image_color/theora'
    sub_img = message_filters.Subscriber(topic, theora_image_transport.msg.Packet)
    sub_img.registerCallback(video_callback)
    print("Entering ROS spin event loop")
    rospy.spin()

if __name__ == '__main__':
    main()
