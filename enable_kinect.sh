#!/bin/bash
function fail() {
    echo "Error starting Kinect"
    echo "Is ROS Indigo installed at /opt/ros/indigo?"
    echo "Is Kinect listed in lsusb?"
    echo "Is ROS Freenect installed? apt-get install libfreenect ros-indigo-freenect-launch"
    exit
}

source /opt/ros/indigo/setup.sh || fail
export ROS_IP=$(ip -4 addr | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | tail -1)

roslaunch freenect_launch freenect.launch
