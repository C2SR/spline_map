import numpy as np
import math
import time

from scipy import sparse
import scipy.sparse.linalg

class SplineMap:
    def __init__(self):
        # Sensor scan parameters
        self.min_angle = 0.
        self.max_angle = 2.*np.pi - 1.*np.pi/180
        self.range_min = 0.12
        self.range_max = 3.5
        # Map parameters
        self.free_detection_per_ray = 15
        # Spline surface parameters
        self.knot_space = .1
        self.degree = 1
        self.xy_min = -10
        self.xy_max = 10
        self.mx = int((self.xy_max - self.xy_min)/self.knot_space)+1
        self.my = int((self.xy_max - self.xy_min)/self.knot_space)+1
        self.ctrl_pts = np.zeros([self.mx,self.my])
        self.time = np.zeros(5) 

    """Removes spurious (out of range) measurements
        Input: ranges np.array<float>
    """ 
    def remove_spurious_measurements(self, ranges):
        # TODO The following two lines are UNnecessarily computed at every iteration
        angles = np.linspace(self.min_angle, self.max_angle, len(ranges) )
        # Finding indices of the valid ranges
        ind = np.logical_and(ranges >= self.range_min, ranges <= self.range_max)
        return ranges[ind], angles[ind] 

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

    """ Detect free space """
    def detect_free_space(self, origin, ranges, angles):
        pts = np.zeros(2).reshape(2,1)
        free_ranges = np.linspace(0,ranges-self.knot_space, self.free_detection_per_ray)
        direction = np.array([np.cos(angles), np.sin(angles)])
        for i in range(0, self.free_detection_per_ray):
            pts_free = free_ranges[i,:] * direction
            pts = np.hstack( (pts, pts_free) )
        return pts
        
    def compute_spline(self, pts_occ, pts_free):
        #pts_occ = np.array([np.arange(-100,100)*.02,np.arange(-100,100)*0.02]) 
        n_occ = pts_occ.shape[1]
        # Computing spline coefficients associated to occupied space
        taux_bar = (pts_occ[0,:]-self.xy_min) % (self.knot_space)
        mux = ((pts_occ[0,:]-self.xy_min)/self.knot_space).astype(int)
        tauy_bar = (pts_occ[1,:]-self.xy_min) % (self.knot_space)        
        muy = ((pts_occ[1,:]-self.xy_min)/self.knot_space).astype(int)

        mx_occ_min = np.min(mux)
        my_occ_min = np.min(muy)
        mx_occ_max = np.max(mux) + 1
        my_occ_max = np.max(muy) + 1   
        mx_occ_len = mx_occ_max - mx_occ_min + 1  
        my_occ_len = my_occ_max - my_occ_min + 1 

        bx0 = (self.knot_space- taux_bar)/self.knot_space 
        bx1 =  taux_bar/self.knot_space                    
        by0 = (self.knot_space- tauy_bar)/self.knot_space 
        by1 =  tauy_bar/self.knot_space                   

        # Kronecker product
        M_occ = sparse.lil_matrix((n_occ, mx_occ_len*my_occ_len))
        index = np.linspace(0, n_occ-1, n_occ).astype(int)       
        M_occ[index,(muy-my_occ_min)*(mx_occ_len)+(mux-mx_occ_min)] = bx0*by0
        M_occ[index,(muy-my_occ_min)*(mx_occ_len)+(mux-mx_occ_min+1)] = bx1*by0
        M_occ[index,(muy-my_occ_min+1)*(mx_occ_len)+(mux-mx_occ_min)] = bx0*by1
        M_occ[index,(muy-my_occ_min+1)*(mx_occ_len)+(mux-mx_occ_min+1)] = bx1*by1

        # ############### Free space #########################
        n_free = pts_free.shape[1]
        # Computing spline coefficients associated to occupied space
        taux_bar = (pts_free[0,:]-self.xy_min) % (self.knot_space)
        mux = ((pts_free[0,:]-self.xy_min)/self.knot_space).astype(int)
        tauy_bar = (pts_free[1,:]-self.xy_min) % (self.knot_space)        
        muy = ((pts_free[1,:]-self.xy_min)/self.knot_space).astype(int)

        mx_free_min = mx_occ_min
        my_free_min = my_occ_min
        mx_free_max = mx_occ_max
        my_free_max = my_occ_max   
        mx_free_len = mx_free_max - mx_free_min + 1  
        my_free_len = my_free_max - my_free_min + 1 

        bx0 = (self.knot_space- taux_bar)/self.knot_space 
        bx1 =  taux_bar/self.knot_space                    
        by0 = (self.knot_space- tauy_bar)/self.knot_space 
        by1 =  tauy_bar/self.knot_space                   

        # Kronecker product
        M_free = sparse.lil_matrix((n_free, mx_free_len*my_free_len))     
        index = np.linspace(0, n_free-1, n_free).astype(int)    
        M_free[index,(muy-my_free_min)*(mx_free_len)+(mux-mx_free_min)] = bx0*by0
        M_free[index,(muy-my_free_min)*(mx_free_len)+(mux-mx_free_min+1)] = bx1*by0
        M_free[index,(muy-my_free_min+1)*(mx_free_len)+(mux-mx_free_min)] = bx0*by1
        M_free[index,(muy-my_free_min+1)*(mx_free_len)+(mux-mx_free_min+1)] = bx1*by1

        # # Fitting the surface using LS      
        P = sparse.eye(mx_occ_len*my_occ_len, format='lil') +  M_occ.T @ M_occ + M_free.T @ M_free
        ctrl_pts = self.ctrl_pts[my_occ_min:my_occ_len+my_occ_min,mx_occ_min:mx_occ_len+mx_occ_min].flatten()
        ctrl_pts= sparse.linalg.spsolve(P.tocsr(), ctrl_pts +  M_occ.T@(M_occ@ctrl_pts+1) +  M_free.T@(M_free@ctrl_pts-1))
 
        ctrl_pts = np.minimum(np.maximum(ctrl_pts,-100),100)
        self.ctrl_pts[my_occ_min:my_occ_len+my_occ_min,
                      mx_occ_min:mx_occ_len+mx_occ_min] = ctrl_pts.reshape(my_occ_len, mx_occ_len)

    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_map(self, pose, ranges):
        # Removing spurious measurements
        tic = time.time()
        ranges, angles = self.remove_spurious_measurements(ranges)
        self.time[0] += time.time() - tic
        # Converting range measurements to metric coordinates
        tic = time.time()
        pts_occ_local = self.range_to_coordinate(ranges, angles)
        self.time[1] += time.time() - tic
        # Detecting free cells in metric coordinates
        tic = time.time()
        pts_free_local = self.detect_free_space(pose[0:2], ranges, angles)
        self.time[2] += time.time() - tic
        # Transforming metric coordinates from the local to the global frame
        tic = time.time()
        pts_occ = self.local_to_global_frame(pose,pts_occ_local)
        pts_free = self.local_to_global_frame(pose,pts_free_local)
        self.time[3] += time.time() - tic
        # Compute spline
        tic = time.time()
        self.compute_spline(pts_occ, pts_free)
        self.time[4] += time.time() - tic