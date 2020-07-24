import numpy as np
from matplotlib import pyplot as plt

from spline_map.spline import SplineLocalization
from spline_map.spline import SplineMap

import sys
import time

def main():
    if len(sys.argv) < 2:
        print("You must enter a file name")
        sys.exit(-1)

    file_name = sys.argv[1]

    # Instantiating the grid map object
    localization = {}
    mapping = {}
    nb_resolution = 4
    for i in range(0,nb_resolution):
        kwargs_spline= {'knot_space': .05*(2**(nb_resolution-i-1)), 
                        'map_size': np.array([25.,25.]),
                        'logodd_occupied': .9,
                        'logodd_free': .7,
                        'nb_iteration_max': 10,
                        'delta_pose_max': 1e-3}
        localization[i] = SplineLocalization(**kwargs_spline)
        mapping[i] = SplineMap(**kwargs_spline)

    # Opening log file
    file_handle = open(file_name, "r")
    # Retrieving sensor parameters
    data = file_handle.readline()  
    data = np.fromstring( data, dtype=np.float, sep=' ' )
    
    # Plot
    fig, axs = plt.subplots(1,max(2,nb_resolution))
    fig.tight_layout()
    plt.show(block=False)

    k=n= 0
    for data in file_handle:
        # collecting the data
        data = np.fromstring( data, dtype=np.float, sep=' ' )
        pose = np.array(data[0:3]) 
        ranges = data[6:]
        for i in range(0, nb_resolution):
            # Localization
            if n < 30:
                localization[i].pose = np.copy(pose)
            else:
                if i==0:
                    pose_estimative = localization[nb_resolution-1].pose 
                else:
                    pose_estimative = localization[i-1].pose 
                localization[i].update_localization(mapping[i], ranges, pose_estimative)
                print(' '.join(map(str, np.hstack([np.array(n), pose, localization[i].pose])    )))
            #update the map
            mapping[i].update_map(localization[i].pose, ranges)
        
        k += 1
        n += 1
        if k > 25 :
            for i in range(0, nb_resolution):
                axs[i].imshow(mapping[i].ctrl_pts.reshape([mapping[i].grid_size[0,0],mapping[i].grid_size[1,0]], order='F'),
                                interpolation='nearest',
                                cmap='gray_r',
                                origin='upper',
                                vmax = mapping[i].logodd_max_occupied, 
                                vmin= mapping[i].logodd_min_free)           
                #ax.set_extent([map.xy_min,map.xy_max-map.knot_space,map.xy_min,map.xy_max-map.knot_space])                               
            plt.pause(.001)
            k = 0    
        if n > 350:
        #    break
            pass
    ## Computing/printing total and per task time
    #localization_total_time = np.sum(localization.time[0:5])
    #localization_avg_time = np.sum(localization_total_time/(n-30))
    #mapping_total_time = np.sum(spline_map.time[0:5])
    #mapping_avg_time = np.sum(mapping_total_time/(n-30))    
    #print('--------')
    #print('Localization avg time: {:.2f} ms'.format(localization_avg_time* 1000) )
    #print('Mapping avg time: {:.2f} ms'.format(mapping_avg_time*1000))
    #print('Diff Localization avg time: {:.2f} ms'.format( 1000*(total_time - localization_total_time)) )  
    #print('--------')
    
    #input("Hit enter to continue")

if __name__ == '__main__':
    main()