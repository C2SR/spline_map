import numpy as np
import math
import time

class SplineLocalization:
    def __init__(self, **kwargs):
        # Parameters
        knot_space = kwargs['knot_space'] if 'knot_space' in kwargs else .05
        min_angle = kwargs['min_angle'] if 'min_angle' in kwargs else 0.
        max_angle = kwargs['max_angle'] if 'max_angle' in kwargs else 2.*np.pi - 1.*np.pi/180.
        angle_increment = kwargs['angle_increment'] if 'angle_increment' in kwargs else 1.*np.pi/180.
        range_min = kwargs['range_min'] if 'range_min' in kwargs else 0.12
        range_max = kwargs['range_max'] if 'range_max' in kwargs else 3.5
        logodd_min_free = kwargs['logodd_min_free'] if 'logodd_min_free' in kwargs else -100
        logodd_max_occupied = kwargs['logodd_max_occupied'] if 'logodd_max_occupied' in kwargs else 100
        det_Hinv_threshold = kwargs['det_Hinv_threshold'] if 'det_Hinv_threshold' in kwargs else 1e-4
        delta_pose_max = kwargs['delta_pose_max'] if 'delta_pose_max' in kwargs else 1e-3
        nb_iteration_max = kwargs['nb_iteration_max'] if 'nb_iteration_max' in kwargs else 10

        # Spline-map parameters
        self.degree = 3
        self.knot_space = knot_space

        # LogOdd Map parameters
        self.logodd_min_free = logodd_min_free
        self.logodd_max_occupied = logodd_max_occupied

        # Sensor scan parameters
        self.min_angle = min_angle
        self.max_angle = max_angle 
        self.angle_increment = angle_increment
        self.range_min = range_min
        self.range_max = range_max
        self.angles = np.arange(min_angle, max_angle+angle_increment, angle_increment )                

        # Localization parameters
        self.delta_pose_max = delta_pose_max
        self.nb_iteration_max = nb_iteration_max        
        self.det_Hinv_threshold = det_Hinv_threshold
        self.pose = np.zeros(3)
        
        # Time
        self.time = np.zeros(5)  

    """Removes spurious (out of range) measurements
        Input: ranges np.array<float>
    """ 
    def remove_spurious_measurements(self, ranges):
        # Finding indices of the valid ranges
        ind_occ = np.logical_and(ranges >= self.range_min, ranges <= self.range_max)
        ranges = np.minimum(np.maximum(ranges, self.range_min), self.range_max)
        return ranges[ind_occ], self.angles[ind_occ]

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
        
        # Spline
        b = np.zeros([nb_pts,self.degree+1])
        b[:,0] = 1/(6)*(-tau_3**3 + 12*tau_3**2 - 48*tau_3 + 64) 
        b[:,1] = 1/(6)*(3*tau_2**3 - 24*tau_2**2 + 60*tau_2 - 44)
        b[:,2] = 1/(6)*(-3*tau_1**3 + 12*tau_1**2 - 12*tau_1 + 4)
        b[:,3] = 1/(6)*(tau_0**3)
        
        # 1st derivative of spline
        db = np.zeros([nb_pts,self.degree+1])
        db[:,0] = 1/(6)*(-3*tau_3**2 + 24*tau_3 - 48 ) 
        db[:,1] = 1/(6)*(9*tau_2**2 - 48*tau_2 + 60 )
        db[:,2] = 1/(6)*(-9*tau_1**2 + 24*tau_1 - 12)
        db[:,3] = 1/(6)*(3*tau_0**2)

        c = np.zeros([nb_pts,(self.degree+1)],dtype='int')
        for i in range(0, self.degree+1):
            c[:,i] = mu-self.degree+i

        return c, b, db

    """"Compute spline tensor coefficients - 2D function """
    def compute_tensor_spline(self, map, pts):
        # Storing number of points
        nb_pts = pts.shape[1]

        # Compute spline along each axis
        cx, bx, dbx  = self.compute_spline(pts[0,:], map.grid_center[0,0])
        cy, by, dby  = self.compute_spline(pts[1,:], map.grid_center[1,0])

        # Compute spline tensor
        ctrl_pt_index = np.zeros([nb_pts,(self.degree+1)**2],dtype='int')
        B = np.zeros([nb_pts,(self.degree+1)**2])
        dB = np.zeros([2*nb_pts,(self.degree+1)**2])
        for i in range(0,self.degree+1):
            for j in range(0,self.degree+1):           
                ctrl_pt_index[:,i*(self.degree+1)+j] = cy[:,i]*(map.grid_size[0,0])+cx[:,j]
                B[:,i*(self.degree+1)+j] = by[:,i]*bx[:,j]
                dB[0::2,i*(self.degree+1)+j] = by[:,i]*dbx[:,j]
                dB[1::2,i*(self.degree+1)+j] = dby[:,i]*bx[:,j]

        return ctrl_pt_index, B, dB

    def compute_pose(self, map, pts_occ_local, pts_occ_global, pose):
        # Spline tensor]
        n_occ = pts_occ_global.shape[1]
        c_index_occ, B_occ, dB_occ = self.compute_tensor_spline(map, pts_occ_global)
        # Current value on the map
        y_est_occ = np.sum(map.ctrl_pts[c_index_occ]*B_occ, axis=1)
        # Fitting error
        e_occ = (1 - y_est_occ/self.logodd_max_occupied) 
        # compute H and b
        dtau = np.array([[1,0,0],[0,1,0]])
        H = np.zeros([3,3])
        b = np.zeros([3,1])
        for i in range(0, n_occ):
            c, s = np.cos(pose[2]), np.sin(pose[2]) 
            dtau[0,2] = -s*pts_occ_local[0,i]-c*pts_occ_local[1,i]
            dtau[1,2] =  c*pts_occ_local[0,i]-s*pts_occ_local[1,i]
            hi = dtau.T @ dB_occ[2*i:2*i+2,:] @ (map.ctrl_pts[c_index_occ[i,:]]).reshape((self.degree+1)**2,1) 
            H += hi @ hi.T 
            b += hi*e_occ[i]

        if np.linalg.det(H) > self.det_Hinv_threshold:
            delta_pose = np.linalg.inv(H)@b
            self.pose[0] += delta_pose[0]
            self.pose[1] += delta_pose[1]
            self.pose[2] += delta_pose[2]
            if self.pose[2] > np.pi:
                self.pose[2] -= 2*np.pi
            elif self.pose[2] < -np.pi:
                self.pose[2] += 2*np.pi
            # giving more weight to orientation
            return np.linalg.norm(np.array([1,1,1.])*delta_pose)
        else:
            print('[Localization] Failed')
            return 1


    """"Occupancy grid mapping routine to update map using range measurements"""
    def update_localization(self, map, ranges, pose_estimative=None):
        if pose_estimative is not None:
            self.pose = pose_estimative
        # Removing spurious measurements
        tic = time.time()
        ranges, angles = self.remove_spurious_measurements(ranges)
        self.time[0] += time.time() - tic
        # Converting range measurements to metric coordinates
        tic = time.time()
        pts_occ_local = self.range_to_coordinate(ranges, angles)
        self.time[1] += time.time() - tic
        residue = 1
        nb_iterations = 0
        while residue > self.delta_pose_max and nb_iterations < self.nb_iteration_max:
            # Transforming metric coordinates from the local to the global frame
            tic = time.time()
            pts_occ = self.local_to_global_frame(self.pose, pts_occ_local)
            self.time[3] += time.time() - tic        
            # Localization
            tic = time.time()
            residue = self.compute_pose(map, pts_occ_local, pts_occ, self.pose)
            self.time[4] += time.time() - tic
            nb_iterations += 1
        return residue, nb_iterations