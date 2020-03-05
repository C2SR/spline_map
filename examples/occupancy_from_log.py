import numpy as np
from matplotlib import pyplot as plt
from spline_map.occupancy import OccupancyGridMap
import sys

def main():
    #TODO: Use better arg handling
    if len(sys.argv) < 2:
        print("You must provide a log file")
        sys.exit(-1)
    #TODO: Error handling for file not being there
    file_name = sys.argv[1]
    # Opening log file
    file_handle = open(file_name, "r")
    
    # Instantiating the grid map object
    kwargs_occupancy_map = {'resolution': .05, 'map_size': np.array([25.,20.])}
    occ_grid_map = OccupancyGridMap(**kwargs_occupancy_map)
    
    # Retrieving sensor parameters
    data = file_handle.readline()  
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    
    # Read the data and build the map
    fig, ax = plt.subplots()
    plt.show(block=False)
    k = 0
    n = 0
    for data in file_handle:
        data = np.fromstring( data, dtype=np.float, sep=' ' )
        pose = data[0:3]
        ranges = data[6:]
        # update the map
        occ_grid_map.update_map(pose, ranges)
        n = n + 1
        k += 1
        if k > 100:
            plt.imshow(occ_grid_map.logodd_map, 
                        interpolation='nearest',
                        cmap='gray_r', 
                        origin='upper',
                        vmax = occ_grid_map.logodd_max_occupied,
                        vmin= occ_grid_map.logodd_min_free)
            plt.pause(.001)
            k = 0
    
    total_time = np.sum(occ_grid_map.time)
    avg_time = np.sum(occ_grid_map.time/n)
    input("Press return key to exit")

if __name__ == '__main__':
    main()
