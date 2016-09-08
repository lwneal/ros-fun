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
from model import Model

import keras


model = None
pubby = None


def init_model():
    global model
    model = Model('model.h5', batch_size=1)

def run_model(image_filename):
    img = Image.open(image_filename)
    img.load()
    question = 'Find the person'
    return model.evaluate(img, question)


def jpg_to_numpy(jpg_data):
    with open('/tmp/robot.jpg', 'w') as fp:
        fp.write(jpg_data)
    return np.array(Image.open('/tmp/robot.jpg'))


def image_callback(img):
    start_time = time.time()

    #img_data = jpg_to_numpy(img.data)
    #save_image(img_data, 'static/kinect.jpg', format="JPEG")
    with open('static/kinect.jpg', 'w') as fp:
        fp.write(img.data)

    # Save one frame per second to the hard drive
    filename = 'turtlebot_data/robot_{}.jpg'.format(int(time.time()))
    with open(filename, 'w') as fp:
        fp.write(img.data)

    convnet_output = run_model('static/kinect.jpg')

    face_detector = np.zeros((32*16, 16*14))
    for i in range(0, 32):
        #print "Face detector shape is {}".format(face_detector.shape)
        outputs = convnet_output['conv5_1'][0]
        #print "outputs shape {}".format(outputs.shape)
        row = np.concatenate(outputs[i*16:(i+1)*16], axis=1)
        #print "row shape is {}".format(row.shape)
        face_detector[i*14:(i+1)*14, :] = row

    #print "img_data shape is {}".format(img_data.shape)
    #print "face_detector shape is {}".format(face_detector.shape)

    img_data = jpg_to_numpy(img.data)
    x = img_data.shape
    y = face_detector.shape
    height = max(x[0], y[0])
    width = x[1] + y[1]
    visual = np.zeros((height, width))
    visual[:x[0], :x[1]] = img_data.mean(axis=2)
    visual[:y[0], x[1]:] = face_detector

    from scipy.misc import imresize
    face_neuron = convnet_output['conv5_1'][0][16*6 + 8]
    face_display = imresize(face_neuron, 8.0, interp='bicubic')

    visual[:face_display.shape[0], :face_display.shape[1]] = face_display
    save_image(visual, 'static/visualize.jpg', format='JPEG')

    handle_movement(face_neuron, face_display)
    #print "Processed frame in {:.2f}s".format(time.time() - start_time)


def handle_movement(face_neuron, face_display):
    # Move the Turtlebot left or right to look at the face
    face_x, face_y = face_position(face_display)
    face_detected = face_neuron.max() > 0
    twist = Twist()
    if face_detected:
        if face_x < -3:
            print "move left"
            twist.linear.x = 0.0
            twist.angular.z = 1.0
            twist.linear.y = 0.0
        elif face_x > 3:
            print("move right")
            twist.linear.x = 0.0
            twist.angular.z = -1.0
            twist.linear.y = 0.0
        else:
            print("move forward")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            twist.linear.y = 0.0
        #pubby.publish(twist)


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
    return to_rel_degrees(x_pos, map_width, fov_width_deg), to_rel_degrees(y_pos, map_height, fov_height_deg)


def save_image(img_data, filename, format="PNG"):
    frame = Image.fromarray(img_data).convert('RGB')
    tmp_name = filename + '.tmp'
    frame.save(tmp_name, format=format)
    os.rename(tmp_name, filename)


def main():
    init_model()
    rospy.init_node('kinect_image_grabber')
    subby = rospy.Subscriber('/camera/rgb/image_color/compressed', sensor_msgs.msg.CompressedImage, image_callback, queue_size=1)
    #subby2 = rospy.Subscriber('/cur_tilt_angle', std_msgs.msg.Float64, position_callback)
    global pubby
    TURTLEBOT_MOVE_TOPIC = '/mobile_base/commands/velocity'
    pubby = rospy.Publisher(TURTLEBOT_MOVE_TOPIC, Twist, queue_size=1)
    print("Entering ROS spin event loop")
    rospy.spin()

if __name__ == '__main__':
    main()
