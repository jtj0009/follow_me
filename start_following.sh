#!/usr/bin/env bash
# File: /home/pi/start_motor.sh

# 1) Source your ROS 2 setup
source /home/pi/ros2_jazzy/install/setup.bash

# 2) Run your Rust motor control binary
# Adjust the path to wherever cargo placed your compiled binary
exec /home/pi/Documents/follow_me/target/release/follow_me
