import numpy as np
import math
import time
class SplineMap:
    def __init__(self):
        # Sensor scan parameters
        self.min_angle = 0.
        self.max_angle = 2.*np.pi - 1.*np.pi/180
        self.range_min = 0.12
        self.range_max = 3.5
        # Map parameters
        self.free_detection_spacing = .5
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
        pts = origin.reshape(2,1)
        for i in range(0, len(ranges)):
            free_range = np.arange(0, ranges[i], self.free_detection_spacing)
            pts_free = np.vstack((free_range * np.cos(angles[i]), free_range * np.sin(angles[i])))
            pts = np.hstack( (pts, pts_free) )
        return pts

    def compute_spline_coefficients(self, tau):
        tau_bar = tau % (self.knot_space)
        if (tau < 0):
            mu = math.trunc(tau/self.knot_space -1)
        else: 
            mu = math.trunc(tau/self.knot_space)            
        b0 = (self.knot_space- tau_bar)/self.knot_space  # (t[mu+1]-tau)/(t[mu+1] - t[mu])
        b1 =  tau_bar/self.knot_space                    # (tau-t[mu])/(t[mu+1] - t[mu])
        return mu, np.array([b0, b1])
        
    def compute_spline(self, pts_occ, pts_free):
        n_occ = pts_occ.shape[1]
        M_occ = np.zeros([n_occ,self.mx*self.my]) # np.zeros([m1*m2,m1*m2])
        phi = np.zeros([self.mx])
        psi = np.zeros([self.my])

        for i in range(0, n_occ):
            mux, Bx = self.compute_spline_coefficients(pts_occ[0,i])    
            muy, By = self.compute_spline_coefficients(pts_occ[1,i])        
            phi[mux+self.mux_origin:mux+self.mux_origin+self.degree+1] = Bx
            psi[muy+self.muy_origin:muy+self.muy_origin+self.degree+1] = By
            M_occ[i,:] = np.kron(phi, psi)
            phi[mux+self.mux_origin:mux+self.mux_origin+self.degree+1] = np.array([0,0])
            psi[muy+self.muy_origin:muy+self.muy_origin+self.degree+1] = np.array([0,0])

        n_free = pts_free.shape[1]
        M_free = np.zeros([n_free,self.mx*self.my]) 
        for i in range(0, n_free):
            mux, Bx = self.compute_spline_coefficients(pts_free[0,i])    
            muy, By = self.compute_spline_coefficients(pts_free[1,i])        
            phi[mux+self.mux_origin:mux+self.mux_origin+self.degree+1] = Bx
            psi[muy+self.muy_origin:muy+self.muy_origin+self.degree+1] = By          
            M_free[i,:] = np.kron(phi, psi)
            phi[mux+self.mux_origin:mux+self.mux_origin+self.degree+1] = np.array([0,0])
            psi[muy+self.muy_origin:muy+self.muy_origin+self.degree+1] = np.array([0,0])             

        A_bar = np.vstack((np.eye(self.mx*self.my), M_occ, M_free))
        b_bar = np.hstack((self.ctrl_pts.flatten(), M_occ@self.ctrl_pts.flatten()+1,M_free@self.ctrl_pts.flatten()-1))
        new_ctrl_pts = np.linalg.lstsq(A_bar, b_bar, rcond=None)[0]
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
        print('[time] Compute spline: ', after-before, ' ms')
        print('--------------------------')