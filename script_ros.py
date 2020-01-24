#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from spline_map import SplineMap
import time

import rosbag
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R

# Instantiating the grid map object
map = SplineMap()
# Opening log file
#bag = rosbag.Bag("linear_motion.bag")
bag = rosbag.Bag("rotational_motion.bag")

poses = [x for x in bag.read_messages(topics=["/base_pose_ground_truth"])]
scans = [x for x in bag.read_messages(topics=["/scan"])]

map.min_angle = scans[0].message.angle_min
map.max_angle = scans[0].message.angle_max

k = 0
n = 0

avg_time = 0
before = time.time()
# Plot
fig, ax = plt.subplots()
plt.show(block=False)
for i in range(min(len(poses),len(scans))):
    position = poses[i].message.pose.pose.position
    orientation = poses[i].message.pose.pose.orientation
    yaw = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_euler('zyx')[0]
    pose = np.array([position.x, position.y, yaw])
    ranges = np.array(list(scans[i].message.ranges))

    map.update_map(pose, ranges)
    avg_time += time.time() - before
    before = time.time()
    k += 1
    n += 1
    if k > 10:
        plt.imshow(map.ctrl_pts.reshape(map.mx,map.my), interpolation='nearest',cmap='gray_r', vmax = 100, vmin=-100)
        plt.pause(.001)
        k = 0   
    print('Average time for spline map after ', n ,' iterations: ', avg_time/n, ' ms')

total_time = np.sum(map.time)
avg_time = np.sum(map.time/n)
print('--------')
print('Removing spurious measurements: {:.2f} ms. Relative time: {:.2f}'.format(map.time[0]/n * 1000, map.time[0]/total_time*100)) 
print('Converting range to coordinate: {:.2f} ms. Relative time: {:.2f}'.format(map.time[1]/n * 1000, map.time[1]/total_time*100)) 
print('Detecting free cells: {:.2f} ms. Relative time: {:.2f}'.format(map.time[2]/n * 1000, map.time[2]/total_time*100)) 
print('Transforming local to global frame: {:.2f} ms. Relative time: {:.2f}'.format(map.time[3]/n * 1000, map.time[3]/total_time*100)) 
print('Updating logodd SPLINE map: {:.2f} ms. Relative time: {:.2f}'.format(map.time[4]/n * 1000, map.time[4]/total_time*100)) 

print('--------')
print('Average time: {:.2f} ms'.format(np.sum(map.time/n) * 1000))
print('Average frequency: {:.2f} Hz'.format(1/(np.sum(map.time/n))))
