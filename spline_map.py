import numpy as np
import math
import time
import scipy.sparse as sps
import scipy.sparse.linalg as spsl

class SplineMap:
    def __init__(self):
        # Sensor scan parameters
        self.min_angle = 0.
        self.max_angle = 2.*np.pi - 1.*np.pi/180
        self.range_min = 0.12
        self.range_max = 3.5
        # Map parameters
        self.free_detection_per_ray = 5
        # Spline surface parameters
        self.knot_space = .5
        self.degree = 1
        self.xy_min = -4.5
        self.xy_max = 4.5
        self.mx = int((self.xy_max - self.xy_min)/self.knot_space)+1
        self.my = int((self.xy_max - self.xy_min)/self.knot_space)+1
        self.mux_origin = int((self.mx-1)/2) 
        self.muy_origin = int((self.my-1)/2) 
        self.ctrl_pts = np.zeros((self.mx,self.my))
        
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
        free_ranges = np.linspace(0,ranges, self.free_detection_per_ray)
        direction = np.array([np.cos(angles), np.sin(angles)])
        for i in range(0, self.free_detection_per_ray):
            pts_free = free_ranges[i,:] * direction
            pts = np.hstack( (pts, pts_free) )
        return pts
        
    def compute_spline(self, pts_occ, pts_free):
        n_occ = pts_occ.shape[1]
        M_occ = np.zeros([n_occ,self.mx*self.my]) # np.zeros([m1*m2,m1*m2])        
        # Computing spline coefficients associated to occupied space
        taux_bar = (pts_occ[0,:]+self.mux_origin) % (self.knot_space)
        mux = (pts_occ[0,:]/self.knot_space+self.mux_origin).astype(int)
        tauy_bar = (pts_occ[1,:]+self.muy_origin) % (self.knot_space)        
        muy = (pts_occ[1,:]/self.knot_space+self.muy_origin).astype(int)
        bx0 = (self.knot_space- taux_bar)/self.knot_space 
        bx1 =  taux_bar/self.knot_space                    
        by0 = (self.knot_space- tauy_bar)/self.knot_space 
        by1 =  tauy_bar/self.knot_space                                    
        index = np.linspace(0, n_occ-1, n_occ).astype(int)
        # Kronecker product
        M_occ[index,mux*self.mx+muy] = bx0*by0
        M_occ[index,mux*self.mx+(muy+1)] = bx0*by1
        M_occ[index,(mux+1)*self.mx+muy] = bx1*by0
        M_occ[index,(mux+1)*self.mx+(muy+1)] = bx1*by1

        # Computing spline coefficients associated to free space
        n_free = pts_free.shape[1]
        M_free = np.zeros([n_free,self.mx*self.my]) 
        taux_bar = (pts_free[0,:]+self.mux_origin) % (self.knot_space)
        mux = (pts_free[0,:]/self.knot_space+self.mux_origin).astype(int)
        tauy_bar = (pts_free[1,:]+self.muy_origin) % (self.knot_space)        
        muy = (pts_free[1,:]/self.knot_space+self.muy_origin).astype(int)
        bx0 = (self.knot_space- taux_bar)/self.knot_space #(t[mu+1]-tau)/(t[mu+1] - t[mu])
        bx1 =  taux_bar/self.knot_space                    #(tau-t[mu])/(t[mu+1] - t[mu])
        by0 = (self.knot_space- tauy_bar)/self.knot_space #(t[mu+1]-tau)/(t[mu+1] - t[mu])
        by1 =  tauy_bar/self.knot_space                    #(tau-t[mu])/(t[mu+1] - t[mu])                
        index = np.linspace(0, n_free-1, n_free).astype(int)
        # Kronecker product
        M_free[index,mux*self.mx+muy] = bx0*by0
        M_free[index,mux*self.mx+(muy+1)] = bx0*by1
        M_free[index,(mux+1)*self.mx+muy] = bx1*by0
        M_free[index,(mux+1)*self.mx+(muy+1)] = bx1*by1      

        # Fitting the surface using LS
        P = np.eye(self.mx*self.my) + M_occ.T @ M_occ + M_free.T @ M_free       
        #sparsity = 1.0 - ( np.count_nonzero(P) / float(P.size) )
        #print('Sparsity of P: ', sparsity)
        P_inv = np.linalg.inv(P)
        new_ctrl_pts = P_inv @ (self.ctrl_pts.flatten() + 
                                            M_occ.T@(M_occ@self.ctrl_pts.flatten()+1 ) + 
                                            M_free.T@(M_free@self.ctrl_pts.flatten()-1 )) 
        new_ctrl_pts = np.minimum(np.maximum(new_ctrl_pts,-100),100)
        self.ctrl_pts = new_ctrl_pts.reshape(self.mx,self.my)

    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_map(self, pose, ranges):
        # Removing spurious measurements
        ranges, angles = self.remove_spurious_measurements(ranges)
        # Converting range measurements to metric coordinates
        pts_occ_local = self.range_to_coordinate(ranges, angles)
        # Detecting free cells in metric coordinates
        before = time.time()
        pts_free_local = self.detect_free_space(pose[0:2], ranges, angles)
        after = time.time() 
        print('[time] Detect free cells: ', after-before, ' ms')
        # Transforming metric coordinates from the local to the global frame
        before = time.time()
        pts_occ = self.local_to_global_frame(pose,pts_occ_local)
        pts_free = self.local_to_global_frame(pose,pts_free_local)
        after = time.time() 
        print('[time] change frame: ', after-before, ' ms')       
        # Compute spline
        before = time.time()
        self.compute_spline(pts_occ, pts_free)
        after = time.time() 
        print('[time] Compute spline: ', after-before), ' ms'
        print('--------------------------')