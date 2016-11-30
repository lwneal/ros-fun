import roslib
#roslib.load_manifest('pr2_position_scripts')

import rospy
import actionlib
from actionlib_msgs.msg import *
from pr2_controllers_msgs.msg import *
from geometry_msgs.msg import *
import message_filters

import random
import sys
import os
import time
import tempfile
import numpy as np
from PIL import Image
import rospy
import roslib
import std_msgs.msg
import sensor_msgs.msg
from geometry_msgs.msg import Twist
from pr2_controllers_msgs.msg import *
from pr2_controllers_msgs.msg import JointTrajectoryControllerState

import joint_states
from model import Model

import keras


g = PointHeadGoal()
g.target.point.x = 1.0
g.target.point.z = .0
g.target.point.y = .0
g.max_velocity = 0.1
#g.min_duration = rospy.Duration(1.0)
g.target.header.frame_id = 'base_link'
#g.target.header.frame_id = 'wide_stereo_l_stereo_camera_frame'

model = None
head_client = None

timestamp = 1

joint_states.init()

def init_model():
    global model
    model = Model('model.h5', batch_size=1)

def run_model(image_filename, command):
    img = Image.open(image_filename)
    img.load()
    return model.evaluate(img, command)


def draw_text(image_array, text):
    from PIL import Image
    from PIL import ImageFont
    from PIL import ImageDraw
    image_array = image_array.astype(np.uint8)
    img = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("sans-serif.ttf", 16)
    #draw.rectangle((0, 0, img.width, 16), fill=(0,0,0,128))
    draw.rectangle((0, 0, img.width, 16), fill=(0,0,0,0))
    draw.text((0, 0), text, (255,255,255), font=font)
    return np.array(img)


def jpg_to_numpy(jpg_data):
    with open('/tmp/robot.jpg', 'w') as fp:
        fp.write(jpg_data)
    return np.array(Image.open('/tmp/robot.jpg'))

def joints_callback(joints):
    ts = joints.header.stamp.to_sec()
    #print "Got callback for joint positions from {:.2f} seconds ago".format(time.time() -ts)

last_move_time = 0
def image_callback(img):
    ts = img.header.stamp.to_sec()
    #print "Got callback for image from {:.2f} seconds ago".format(time.time() - ts)

def master_callback(joints, img):
    joints_ts = joints.header.stamp.to_sec()
    img_ts = img.header.stamp.to_sec()
    print "Got MASTER callback with image {:.2f}s and joints {:.2f}s old".format(
            time.time() - img_ts, time.time() - joints_ts)
    global last_move_time
    global timestamp
    start_time = time.time()

    img_timestamp = img.header.stamp.to_sec()

    #img_data = jpg_to_numpy(img.data)
    #save_image(img_data, 'static/kinect.jpg', format="JPEG")
    with open('static/kinect.jpg', 'w') as fp:
        fp.write(img.data)

    img = np.array(Image.open('static/kinect.jpg').convert('RGB'))
    #print "Processing image from {:.3f} seconds ago".format(time.time() - img_timestamp)

    command = open('static/question.txt').read()
    model_output = run_model('static/kinect.jpg', command)
    height, width = model_output.shape

    from scipy.misc import imresize
    visual = 0.5 * imresize(img, (480, 640))

    # Output final layer activations as the red channel
    overlay = imresize(model_output, (480, 640), interp='nearest')

    # De-normalize so we don't scale small probabilities up to red
    overlay = (overlay.astype(np.float) * model_output.max()).astype(np.int)

    #np.where(visual[:,:,0] > .05) = np.clip(0, 255, overlay * 1000)
    visual[:,:,0] = (visual[:,:,0] + overlay).clip(0,255)


    # Output a minimap overlay of the model output
    visual[380:, 540:, 0] = imresize(model_output, (100,100))
    visual[380:, 540:, 1] = imresize(model_output, (100,100))
    visual[380:, 540:, 2] = imresize(model_output, (100,100))


    MOVEMENT_DELAY = .01
    if last_move_time + MOVEMENT_DELAY < time.time():
        azumith, altitude, certainty = handle_movement(model_output)
        last_move_time = time.time()

    visual = draw_text(visual, "Command: '{}'\nDetected target with certainty {:.2f} at azumith {:.2f} deg altitude {:.2f}".format(
        command, certainty, azumith, altitude))

    save_image(visual, 'static/visual.jpg', format='JPEG')

    # Save frame to the hard drive
    VIDEO_DIR = 'data/recording'
    #timestamp = '{}.jpg'.format(int(time.time() * 1000))
    timestamp += 1
    filename = os.path.join(VIDEO_DIR, '{}.jpg'.format(timestamp))
    save_image(visual, filename)

    #print "Processed frame in {:.2f}s".format(time.time() - start_time)

    if '--visual' in sys.argv:
        # Draw image
        os.system('./imgcat static/visual.jpg')


last_move_at = 0
def handle_movement(model_output):
    global last_move_at
    global g
    # Move the Turtlebot left or right to look at the brightest pixel
    azumith, altitude = face_position(model_output)
    certainty = model_output.max()
    #print("Target detected with certainty {:.2f} at azumith {:.2f}".format(certainty, azumith))

    SPEED = .01
    if certainty > .01 and time.time() - last_move_at > 1.0:
        g.target.point.x = 2.0
        g.target.point.z = 1.5 - SPEED * altitude
        g.target.point.y = -SPEED * azumith
        head_client.send_goal(g)
        print("Current pos {} {}".format(joint_states.pan, joint_states.tilt))
        last_move_at = time.time()

    return azumith, altitude, certainty


def face_position(convnet_map, fov_width_deg=62.0, fov_height_deg=48.6):
    """ Inputs: 
    Numpy 2D array (eg. VGG16 layer conv5_1, filter 104)
    Field of View (eg. Kinect 1 FOV is 62x48.6)
    Outputs:
    Horizontal position: Degrees, negative is left, zero is straight ahead
    Vertical position: Degrees, negative is up, zero is straight ahead
    """
    map_height, map_width = convnet_map.shape
    y_pos = np.argmax(np.sum(convnet_map, axis=1), axis=0)
    x_pos = np.argmax(np.sum(convnet_map, axis=0))

    def to_rel_degrees(position_idx, map_length, fov_degrees):
        return (position_idx * (1.0 / map_length) - 0.5) * fov_degrees

    # Get the weighted average of columns
    idxs = np.arange(0, map_width)
    col_avg = np.average(idxs, weights=np.sum(convnet_map, axis=0))

    return to_rel_degrees(col_avg, map_width, fov_width_deg), to_rel_degrees(y_pos, map_height, fov_height_deg)


def save_image(img_data, filename, format="PNG"):
    frame = Image.fromarray(img_data).convert('RGB')
    tmp_name = filename + '.tmp'
    frame.save(tmp_name, format=format)
    os.rename(tmp_name, filename)


def main():
    global head_client
    init_model()
    rospy.init_node('remote_gpu_control')

    # Take control of the head
    head_client = actionlib.SimpleActionClient('/head_traj_controller/point_head_action', PointHeadAction)
    # Move the head to a default position
    head_client.send_goal(g)
    head_client.wait_for_server()

    # Subscribe to the video feed (JPG stream)
    camera = '/wide_stereo/left/image_color/compressed'
    #sub_img = rospy.Subscriber(camera, sensor_msgs.msg.CompressedImage, image_callback, queue_size=1)
    sub_img = message_filters.Subscriber(camera, sensor_msgs.msg.CompressedImage)
    sub_img.registerCallback(image_callback)

    # Subscribe to the telemetry feed
    joint_topic = '/joint_states'
    #sub_joints = rospy.Subscriber(joint_topic, sensor_msgs.msg.JointState, joints_callback, queue_size=1)
    sub_joints = message_filters.Subscriber(joint_topic, sensor_msgs.msg.JointState)
    sub_joints.registerCallback(joints_callback)

    sync = message_filters.ApproximateTimeSynchronizer([sub_joints, sub_img], 6, .5)
    sync.registerCallback(master_callback)

    print("Entering ROS spin event loop")
    rospy.spin()

if __name__ == '__main__':
    main()
