import numpy as np
import math
import time

import scipy.sparse.linalg

class SplineMap:
    def __init__(self, **kwargs):
        # Parameters
        knot_space = kwargs['knot_space'] if 'knot_space' in kwargs else .05
        map_size = kwargs['map_size'] if 'map_size' in kwargs else np.array([10.,10.]) 
        min_angle = kwargs['min_angle'] if 'min_angle' in kwargs else 0.
        max_angle = kwargs['max_angle'] if 'max_angle' in kwargs else 2.*np.pi - 1.*np.pi/180.
        angle_increment = kwargs['angle_increment'] if 'angle_increment' in kwargs else 1.*np.pi/180.
        range_min = kwargs['range_min'] if 'range_min' in kwargs else 0.12
        range_max = kwargs['range_max'] if 'range_max' in kwargs else 3.6
        logodd_occupied = kwargs['logodd_occupied'] if 'logodd_occupied' in kwargs else .9
        logodd_free = kwargs['logodd_free'] if 'logodd_free' in kwargs else .3
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100

        # Spline-map parameters
        # @TODO grid_size has to be greater than (2d x 2d)
        self.degree = 3
        self.knot_space = knot_space
        self.grid_size = np.ceil(map_size/knot_space+self.degree).astype(int).reshape([2,1]) 
        self.grid_center = np.ceil((self.grid_size-self.degree)/2).reshape(2,1) + self.degree - 1  
        self.ctrl_pts = .5*(logodd_max_occupied+logodd_min_free)*np.ones((self.grid_size[0,0], self.grid_size[1,0]) ).flatten()

        # Map parameters
        self.map_increment = range_max    
        self.map_lower_limits = (self.degree - self.grid_center)*self.knot_space
        self.map_upper_limits = (self.grid_size-self.grid_center+1)*self.knot_space          

        # LogOdd Map parameters
        self.logodd_occupied = logodd_occupied
        self.logodd_free = logodd_free
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied
        self.free_detection_spacing = 1.41*knot_space 
        self.free_ranges = np.arange(min(knot_space, range_min), range_max, self.free_detection_spacing)        
        
        # Sensor scan parameters
        self.min_angle = min_angle
        self.max_angle = max_angle 
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.angles = np.arange(min_angle, max_angle+angle_increment, angle_increment )

        # Time
        self.time = np.zeros(5)           

    """Removes spurious (out of range) measurements
        Input: ranges np.array<float>
    """ 
    def remove_spurious_measurements(self, ranges):
        # Finding indices of the valid ranges
        ind_occ = np.logical_and(ranges >= self.range_min, ranges <= self.range_max)
        ind_free = (ranges >= self.range_min)        
        ranges = np.minimum(np.maximum(ranges, self.range_min), self.range_max)
        return ranges[ind_occ], self.angles[ind_occ], ranges[ind_free], self.angles[ind_free] 

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

    """Resize the map"""
    def update_map_size(self, pts):
        max_pts_coord = np.max(pts,axis=1).reshape(2,1)
        min_pts_coord = np.min(pts,axis=1).reshape(2,1)        

        if ((max_pts_coord >= self.map_upper_limits).any() or (min_pts_coord < self.map_lower_limits).any()):
            print('resizing the map..')
            # Check if most POSITIVE coordinates are out of bounds
            is_pts_outside_map = (max_pts_coord >= self.map_upper_limits)
            pos_map_increment = is_pts_outside_map * self.map_increment         
            # Check if most NEGATIVE coordinates are out of bounds          
            is_pts_outside_map = (min_pts_coord < self.map_lower_limits)
            neg_map_increment = is_pts_outside_map * self.map_increment   
            # Create new ctrl map map and copy previous map
            new_map_size = self.map_upper_limits - self.map_lower_limits + pos_map_increment + neg_map_increment 
            new_grid_size = np.ceil(new_map_size/self.knot_space+self.degree).astype(int).reshape([2,1]) 
            neg_grid_size_increment = (neg_map_increment/self.knot_space).astype(int)
            new_ctrl_pts = np.zeros([new_grid_size[0,0], new_grid_size[1,0]])
            new_ctrl_pts[neg_grid_size_increment[0,0]:self.grid_size[0,0]+neg_grid_size_increment[0,0],
                    neg_grid_size_increment[1,0]:self.grid_size[1,0]+neg_grid_size_increment[1,0]] = self.ctrl_pts.reshape([self.grid_size[0,0],self.grid_size[1,0]], order='F')

            self.ctrl_pts = new_ctrl_pts.flatten(order='F')
            self.grid_size = new_grid_size
            self.grid_center += neg_grid_size_increment            
            self.map_lower_limits = (self.degree - self.grid_center)*self.knot_space
            self.map_upper_limits = (self.grid_size-self.grid_center+1)*self.knot_space  

    """ Detect free space """
    def detect_free_space(self, origin, ranges, angles):
        pts = np.zeros([2,2])#.reshape(2,1)
        direction = np.array([np.cos(angles), np.sin(angles)])
        for i in range(0, len(self.free_ranges)):
            pts_free = (self.free_ranges[i]) * direction[:,ranges - 1.41*self.knot_space > self.free_ranges[i]]
            pts = np.hstack( (pts, pts_free) )
        # @TODO Test if this is faster when len(ranges) < len(self.free_ranges) 
        #for i in range(0, len(ranges)):
        #    pts_free = direction[:,i].reshape([2,1]) * (self.free_ranges[self.free_ranges <= ranges[i] - 1.41*self.knot_space]) 
        #    pts = np.hstack( (pts, pts_free) )
        return pts

    """"Compute spline coefficients - 1D function """
    def compute_spline(self, tau, origin):
        # Number of points
        nb_pts = len(tau)
        # Normalize regressor
        mu    = -(np.ceil(-tau/self.knot_space).astype(int)) + origin
        tau_bar = (tau/self.knot_space + origin) % 1 

        # Compute spline function along the x-axis        
        tau_3 = tau_bar + 3
        tau_2 = tau_bar + 2        
        tau_1 = tau_bar + 1
        tau_0 = tau_bar
        
        b = np.zeros([nb_pts,self.degree+1])
        b[:,0] = 1/(6)*(-tau_3**3 + 12*tau_3**2 - 48*tau_3 + 64) 
        b[:,1] = 1/(6)*(3*tau_2**3 - 24*tau_2**2 + 60*tau_2 - 44)
        b[:,2] = 1/(6)*(-3*tau_1**3 + 12*tau_1**2 - 12*tau_1 + 4)
        b[:,3] = 1/(6)*(tau_0**3)

        c = np.zeros([nb_pts,(self.degree+1)],dtype='int')
        for i in range(0, self.degree+1):
            c[:,i] = mu-self.degree+i

        return b, c

    """"Compute spline tensor coefficients - 2D function """
    def compute_tensor_spline(self, pts):
        # Storing number of points
        nb_pts = pts.shape[1]

        # Compute spline along each axis
        bx, cx = self.compute_spline(pts[0,:], self.grid_center[0,0])
        by, cy = self.compute_spline(pts[1,:], self.grid_center[1,0])

        # Compute spline tensor
        B = np.zeros([nb_pts,(self.degree+1)**2])
        for i in range(0,self.degree+1):
            for j in range(0,self.degree+1):           
                B[:,i*(self.degree+1)+j] = by[:,i]*bx[:,j]

        # Kronecker product for index
        ctrl_pt_index = np.zeros([nb_pts,(self.degree+1)**2],dtype='int')
        for i in range(0, self.degree+1):
            for j in range(0, self.degree+1):
                ctrl_pt_index[:,i*(self.degree+1)+j] = cy[:,i]*(self.grid_size[0,0])+cx[:,j]

        return B, ctrl_pt_index

    """"Update the control points of the spline map"""
    def update_spline_map(self, pts_occ, pts_free):
        # Storing number of points
        n_occ = pts_occ.shape[1]
        n_free = pts_free.shape[1]

        # Computing spline tensor
        B_occ, c_index_occ = self.compute_tensor_spline(pts_occ)
        B_free, c_index_free = self.compute_tensor_spline(pts_free)

        # Control points index 
        c_index_min = min(np.min(c_index_occ[:,0]), np.min(c_index_free[:,0]))
        c_index_max = max(np.max(c_index_occ[:,-1]), np.max(c_index_free[:,-1]))

        # Current value on the map
        y_est_occ = np.sum(self.ctrl_pts[c_index_occ]*B_occ, axis=1)
        y_est_free = np.sum(self.ctrl_pts[c_index_free]*B_free, axis=1)
        
        # Magnitude of the gradient
        B_occ_norm = np.linalg.norm(B_occ, axis=1)
        B_occ_norm_squared = B_occ_norm**2
        B_free_norm = np.linalg.norm(B_free, axis=1)
        B_free_norm_squared = B_free_norm**2

        # Fitting error
        e_occ = (self.logodd_max_occupied - y_est_occ)      
        mag_occ = np.minimum(self.logodd_occupied/B_occ_norm_squared, np.abs(e_occ)) * np.sign(e_occ)
        e_free = (self.logodd_min_free - y_est_free)      
        mag_free = np.minimum(self.logodd_free/B_free_norm_squared, np.abs(e_free)) * np.sign(e_free)
                
        np.add.at(self.ctrl_pts, c_index_occ, (B_occ.T*mag_occ).T)
        np.add.at(self.ctrl_pts, c_index_free, (B_free.T*mag_free).T)

        # Forcing the points to remain bounded
        self.ctrl_pts[c_index_min:c_index_max+1] = np.minimum(np.maximum(self.ctrl_pts[c_index_min:c_index_max+1], self.logodd_min_free), self.logodd_max_occupied)
            
    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_map(self, pose, ranges):
        # Removing spurious measurements
        tic = time.time()
        ranges, angles, ranges_free, angles_free = self.remove_spurious_measurements(ranges)
        self.time[0] += time.time() - tic
        # Converting range measurements to metric coordinates
        tic = time.time()
        pts_occ_local = self.range_to_coordinate(ranges, angles)
        pts_free_end_local = self.range_to_coordinate(ranges_free, angles_free)
        self.time[1] += time.time() - tic
        # Transforming metric coordinates from the local to the global frame
        tic = time.time()
        pts_occ = self.local_to_global_frame(pose,pts_occ_local)
        pts_free_end = self.local_to_global_frame(pose,pts_free_end_local)
        self.update_map_size(pts_free_end)
        self.time[3] += time.time() - tic        
        # Detecting free cells in metric coordinates
        tic = time.time()
        pts_free_local = self.detect_free_space(pose[0:2], ranges_free, angles_free)
        self.time[2] += time.time() - tic
        tic = time.time()
        pts_free = self.local_to_global_frame(pose,pts_free_local)
        self.time[3] += time.time() - tic
        # Compute spline
        tic = time.time()
        self.update_spline_map(pts_occ, pts_free)
        self.time[4] += time.time() - tic
        
