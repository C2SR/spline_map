import numpy as np
from matplotlib import pyplot as plt
from spline_map import SplineMap
import time

# Instantiating the grid map object
map = SplineMap()
# Opening log file
file_handle = open("laser_log.txt", "r")
# Retrieving sensor parameters
data = file_handle.readline()  
data = np.fromstring( data, dtype=np.float, sep=' ' )
map.min_angle = data[3] 
map.max_angle = data[4]
# Read the data and build the map
k = 10
n = 0
avg_time = 0
# Plot
fig, ax = plt.subplots()
plt.show(block=False)
for data in file_handle:
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    pose = data[0:3]
    ranges = data[6:]
    # update the map
    before = time.time()
    map.update_map(pose, ranges)
    avg_time += time.time() - before
    #print(after-before)
    k += 1
    n += 1
    if k > 10:
        plt.imshow(map.ctrl_pts.reshape(map.mx,map.my), interpolation='nearest',cmap='gray_r', vmax = 100, vmin=-100)
        plt.pause(.001)
        k = 0   
    #print('Average time for spline map after ', n ,' iterations: ', avg_time/n, ' ms')
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