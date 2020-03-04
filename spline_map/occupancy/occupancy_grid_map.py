# Basic libraries
import numpy as np
import time

# Internal libraries
from spline_map.occupancy import bresenham

class OccupancyGridMap:
    """ Occupancy grid class
    resolution: cell resolution in meters 
    map_size: np.array(2) with the (x,y) size of the map in meters
    map_origin: np.array(2) with an offset to the origin of the map in the meters
    """
    def __init__(self, **kwargs): 
        # Parameters
        resolution = kwargs['resolution'] if 'resolution' in kwargs else .1
        map_size = kwargs['map_size'] if 'map_size' in kwargs else np.array([10.,10.]) 
        min_angle = kwargs['min_angle'] if 'min_angle' in kwargs else 0.
        max_angle = kwargs['max_angle'] if 'max_angle' in kwargs else 2.*np.pi - 1.*np.pi/180.
        angle_increment = kwargs['angle_increment'] if 'angle_increment' in kwargs else 1.*np.pi/180.
        range_min = kwargs['range_min'] if 'range_min' in kwargs else 0.12
        range_max = kwargs['range_max'] if 'range_max' in kwargs else 3.5
        logodd_occupied = kwargs['logodd_occupied'] if 'logodd_occupied' in kwargs else .9
        logodd_free = kwargs['logodd_free'] if 'logodd_free' in kwargs else .7
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100

        # Grid parameters
        self.resolution = resolution
        self.grid_size = np.array(map_size/resolution).astype(int).reshape([2,1]) + \
                            (np.array(map_size/resolution).astype(int).reshape([2,1]) % 2)   # these coordinates are always odd
        self.grid_center = np.array((self.grid_size-1)/2, dtype=int).reshape(2,1) 
        
        # LogOdd Map parameters
        self.logodd_map = np.zeros( (self.grid_size[0,0], self.grid_size[1,0]) )
        self.logodd_occupied = logodd_occupied
        self.logodd_free = logodd_free
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied

        # Sensor scan parameters
        self.min_angle = min_angle
        self.max_angle = max_angle 
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max

        # Timing purposes
        self.time = np.zeros(6) 
              
    """Removes spurious (out of range) measurements
        Input: ranges np.array<float>
    """ 
    def remove_spurious_measurements(self, ranges):
        # TODO The following two lines are UNnecessarily computed at every iteration
        
        angles = np.linspace(self.min_angle, self.max_angle, len(ranges) )
        # Finding indices of the valid ranges
        ind_occ = np.logical_and(ranges >= self.range_min, ranges <= self.range_max)
        ind_free = (ranges >= self.range_min)        
        ranges = np.minimum(np.maximum(ranges, self.range_min), self.range_max)
        return ranges[ind_occ], angles[ind_occ], ranges[ind_free], angles[ind_free] 

    """ Transforms ranges measurements to (x,y) coordinates (local frame) """
    def range_to_coordinate(self, ranges, angles):
        angles = np.array([np.cos(angles), np.sin(angles)]) 
        return ranges * angles 

    """ Transform an [2xn] array of (x,y) coordinates to the global frame
        Input: pose np.array<float(3,1)> describes (x,y,theta)'
    """
    def local_to_global_frame(self, pose, local):
        c, s = np.cos(pose[2]), np.sin(pose[2])
        R = np.array([[c, -s],[s, c]])
        return np.matmul(R, local) + pose[0:2].reshape(2,1) 
 
    """Converts metric coordinate to grid coordinate"""
    def metric_to_grid_coordinate(self, pose, map_coordinate):
        pose_grid = np.array(pose[0:2]/self.resolution).reshape([2,1]).astype(int) + self.grid_center
        grid_coordinate = np.array(map_coordinate/self.resolution).astype(int) + self.grid_center
        return pose_grid, grid_coordinate
    
    """Computes free cells using bresenham algorithm""" 
    def compute_free_cells(self, origin, free_cell_end):
        free_cells = origin
        for i in range(0, free_cell_end.shape[1]):
            ray = np.array(bresenham(origin[:,0], free_cell_end[:,i])).T
            free_cells = np.hstack( (free_cells, ray[:,1:-1]) )     # Removing origin and obstacle cells
        return free_cells

    """Updates map following logodd approach"""
    def update_cell_occupancy(self, origin, occupied, free):
        self.logodd_map[origin[0,0], origin[1,0]] = self.logodd_min_free        
        for cell in free.T:
            self.logodd_map[cell[0],cell[1]] = np.maximum(self.logodd_min_free,
                self.logodd_map[cell[0], cell[1]] - self.logodd_free)
        for cell in occupied.T:
            self.logodd_map[cell[0],cell[1]] = np.minimum(self.logodd_max_occupied,
                self.logodd_map[cell[0], cell[1]] + self.logodd_occupied) 

    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_map(self, pose, ranges):
        # Removing spurious measurements
        tic = time.time()
        ranges_occ, angles_occ, ranges_free_end, angles_free_end = self.remove_spurious_measurements(ranges)
        self.time[0] += time.time() - tic
        # Converting range measurements to metric coordinates
        tic = time.time()
        pts_occ_local = self.range_to_coordinate(ranges_occ, angles_occ)
        pts_free_end_local = self.range_to_coordinate(ranges_free_end, angles_free_end)
        self.time[1] += time.time() - tic
        # Transforming metric coordinates from the local to the global frame
        tic = time.time()
        pts_occ_global = self.local_to_global_frame(pose,pts_occ_local)
        pts_free_end_global = self.local_to_global_frame(pose,pts_free_end_local)
        self.time[2] += time.time() - tic
        # Transform metric coordinates to grid cell (integer) coordinates
        tic = time.time()
        pose_cell, occupied_cells = self.metric_to_grid_coordinate(pose, pts_occ_global)
        pose_cell, free_end_cells = self.metric_to_grid_coordinate(pose, pts_free_end_global)
        self.time[3] += time.time() - tic
        # Detected free cells (bresenham algorithms)
        tic = time.time()
        free_cells = self.compute_free_cells(pose_cell, occupied_cells)
        self.time[4] += time.time() - tic
        # Update logodd map 
        tic = time.time()
        self.update_cell_occupancy(pose_cell, occupied_cells, free_cells)
        self.time[5] += time.time() - tic

    


