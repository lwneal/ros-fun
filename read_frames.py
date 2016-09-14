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
from model import Model

import keras


model = None
pubby = None


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
    img = Image.fromarray(image_array).convert('RGB')
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("sans-serif.ttf", 16)
    draw.rectangle((0, 0, img.width, 16), fill=(0,0,0,128))
    draw.text((0, 0), text, (255,255,255), font=font)
    return np.array(img)


def jpg_to_numpy(jpg_data):
    with open('/tmp/robot.jpg', 'w') as fp:
        fp.write(jpg_data)
    return np.array(Image.open('/tmp/robot.jpg'))


last_move_time = 0
def image_callback(img):
    global last_move_time
    start_time = time.time()

    #img_data = jpg_to_numpy(img.data)
    #save_image(img_data, 'static/kinect.jpg', format="JPEG")
    with open('static/kinect.jpg', 'w') as fp:
        fp.write(img.data)

    # Save frame to the hard drive
    VIDEO_DIR = 'data/recording'
    timestamp = '{}.jpg'.format(int(time.time() * 1000))
    filename = os.path.join(VIDEO_DIR, timestamp)
    with open(filename, 'w') as fp:
        fp.write(img.data)

    img = np.array(Image.open('static/kinect.jpg'))

    command = open('static/question.txt').read()
    model_output = run_model('static/kinect.jpg', command)
    height, width = model_output.shape

    from scipy.misc import imresize
    visual = imresize(img, (480, 640))

    # Output final layer activations as the red channel
    visual[:,:,0] = imresize(model_output, (480, 640))

    save_image(visual, 'static/visual.jpg', format='JPEG')

    draw_text(img, "Command: {}".format(command))

    MOVEMENT_DELAY = .1
    if last_move_time + MOVEMENT_DELAY < time.time():
        handle_movement(model_output)
        last_move_time = time.time()
    print "Processed frame in {:.2f}s".format(time.time() - start_time)

    if '--visual' in sys.argv:
        # Draw image
        os.system('./imgcat static/visual.jpg')


def handle_movement(model_output):
    # Move the Turtlebot left or right to look at the brightest pixel
    face_x, face_y = face_position(model_output)
    face_detected = model_output.max() > .5
    twist = Twist()
    if face_detected:
        if abs(face_x) > 5:
            print "move by {}".format(face_x)
            twist.linear.x = 0.0
            twist.angular.z = -.02 * face_x
            twist.linear.y = 0.0
        else:
            print("move forward")
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            twist.linear.y = 0.0
        pubby.publish(twist)
    else:
        print("I don't see anything interesting")


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
