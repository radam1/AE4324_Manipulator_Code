#!/bin/bash

source /opt/ros/jazzy/setup.bash

cd ~/ros2_jazzy/edubot/python_impl/

colcon build --symlink-install
