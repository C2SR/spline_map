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
        self.free_detection_per_ray = 25

        # Spline surface parameters
        self.knot_space = .1
        self.free_detection_spacing = 1.5*self.knot_space 
        self.degree = 1
        self.xy_min = -3.5
        self.xy_max = 3.5
        self.mx = int((self.xy_max - self.xy_min)/self.knot_space)+1
        self.my = int((self.xy_max - self.xy_min)/self.knot_space)+1
        self.ctrl_pts = np.zeros(self.mx*self.my)
        self.time = np.zeros(5) 
        self.free_ranges = np.arange(0, self.range_max, self.free_detection_spacing)
 
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
        direction = np.array([np.cos(angles), np.sin(angles)])
        for i in range(0, len(self.free_ranges)):
            pts_free = self.free_ranges[i] * direction[:,ranges > self.free_ranges[i]]
            pts = np.hstack( (pts, pts_free) )
        return pts

    def compute_spline(self, pts):
        # Computing spline coefficients associated to occupied space
        taux_bar = (pts[0,:]-self.xy_min) % (self.knot_space)
        mux = ((pts[0,:]-self.xy_min)/self.knot_space).astype(int)
        tauy_bar = (pts[1,:]-self.xy_min) % (self.knot_space)        
        muy = ((pts[1,:]-self.xy_min)/self.knot_space).astype(int)

        bx0 = (self.knot_space- taux_bar)/self.knot_space 
        bx1 =  taux_bar/self.knot_space                    
        by0 = (self.knot_space- tauy_bar)/self.knot_space 
        by1 =  tauy_bar/self.knot_space

        # Kronecker product
        c1 = (muy)*(self.mx)+(mux)
        c2 = (muy)*(self.mx)+(mux+1)
        c3 = (muy+1)*(self.mx)+(mux)
        c4 = (muy+1)*(self.mx)+(mux+1)

        return bx0, bx1, by0, by1, c1, c2, c3, c4

    def update_spline_map(self, pts_occ, pts_free):
        # Computing spline coefficients associated to occupied space
        # ############### Occupied space #########################
        n_occ = pts_occ.shape[1]
        bx0_occ, bx1_occ, by0_occ, by1_occ, c1_occ, c2_occ, c3_occ, c4_occ = self.compute_spline(pts_occ)
        occ_cols = np.hstack((c1_occ,c2_occ,c3_occ,c4_occ))
        occ_row_index = np.linspace(0, n_occ-1, n_occ).astype(int) 

        # ############### Free space #########################
        n_free = pts_free.shape[1]
        bx0_free, bx1_free, by0_free, by1_free, c1_free, c2_free, c3_free, c4_free = self.compute_spline(pts_free)        
        free_cols = np.hstack((c1_free,c2_free,c3_free,c4_free))
        free_row_index = np.linspace(0, n_free-1, n_free).astype(int)
         
        # Finding control points that have to be updated 
        cols_unique = np.unique(np.hstack((occ_cols, free_cols)))
        col_index = np.zeros(cols_unique[-1]-cols_unique[0]+1, dtype=int)
        k = 0
        for i in range(cols_unique[0], cols_unique[-1]+1):
            if cols_unique[k] == i:
                col_index[i-cols_unique[0]] = k
                k+=1

        M_occ = sparse.lil_matrix((n_occ, len(cols_unique)))
        M_occ[occ_row_index,col_index[c1_occ-cols_unique[0]]] = bx0_occ*by0_occ
        M_occ[occ_row_index,col_index[c2_occ-cols_unique[0]]] = bx1_occ*by0_occ
        M_occ[occ_row_index,col_index[c3_occ-cols_unique[0]]] = bx0_occ*by1_occ
        M_occ[occ_row_index,col_index[c4_occ-cols_unique[0]]] = bx1_occ*by1_occ

        M_free = sparse.lil_matrix((n_free, len(cols_unique)))
        M_free[free_row_index,col_index[c1_free-cols_unique[0]]] = bx0_free*by0_free
        M_free[free_row_index,col_index[c2_free-cols_unique[0]]] = bx1_free*by0_free
        M_free[free_row_index,col_index[c3_free-cols_unique[0]]] = bx0_free*by1_free
        M_free[free_row_index,col_index[c4_free-cols_unique[0]]] = bx1_free*by1_free

        # # Fitting the surface using LS      
        P =   scipy.sparse.vstack((sparse.eye(len(cols_unique), format='lil'), M_occ, M_free))
        ctrl_pts = self.ctrl_pts[cols_unique]
        b = np.hstack((ctrl_pts, (M_occ@ctrl_pts+1), (M_free@ctrl_pts-1)))
        ctrl_pts= sparse.linalg.lsqr(P.tocsr(),  b)
        ctrl_pts = np.minimum(np.maximum(np.array(ctrl_pts[0]),-100),100)
        self.ctrl_pts[cols_unique] = ctrl_pts

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
        self.update_spline_map(pts_occ, pts_free)
        self.time[4] += time.time() - tic