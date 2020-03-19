#!/usr/bin/env python
import numpy as np

import rosbag
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

def main():
    #TODO: Use better arg handling
    if len(sys.argv) < 2:
        print("You must provide the path to a bag file")
        sys.exit(-1)
    #TODO: Error handling for file not being there
    bag_path = sys.argv[1]

    # Opening log file
    bag = rosbag.Bag(bag_path)

    poses = [x for x in bag.read_messages(topics=["/odom"])]
    scans = [x for x in bag.read_messages(topics=["/scan"])]

    if (len(poses) != len(scans)):
        print("[WARN] Different number of odom and scan messages!")

    for i in range(min(len(poses),len(scans))):
        # Extracting pose from odometry message
        position = poses[i].message.pose.pose.position
        orientation = poses[i].message.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        # Scan messages
        ranges = np.array(list(scans[i].message.ranges))
        zero_padding = np.zeros(3)
        print position.x, position.y, yaw, ' '.join(map(str, zero_padding)), ' '.join(map(str, ranges))