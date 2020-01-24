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
        self.knot_space = .05
        self.free_detection_spacing =  1.25*self.knot_space 
        self.degree = 2
        self.xy_min = -3.5
        self.xy_max = 3.5
        self.mx = int((self.xy_max - self.xy_min)/self.knot_space) + self.degree
        self.my = int((self.xy_max - self.xy_min)/self.knot_space) + self.degree
        self.ctrl_pts = 50*np.ones(self.mx*self.my)
        self.time = np.zeros(7) 
        self.free_ranges = np.arange(0, self.range_max, self.free_detection_spacing)
 
        # Vizualization
        x = np.arange(self.xy_min, self.xy_max- self.knot_space, self.knot_space/2) 
        y = np.arange(self.xy_min, self.xy_max- self.knot_space, self.knot_space/2)
        x, y = np.meshgrid(x,y)
        pts = np.vstack((x.flatten(), y.flatten()))
        bx0, bx1, bx2, by0, by1, by2, c1, c2, c3, c4, c5, c6, c7, c8, c9 = self.compute_spline(pts)
        n = pts.shape[1]
        row_index = np.linspace(0, n-1, n).astype(int) 
        self.plot_size_x = len(x)
        self.plot_size_y = len(y)        
        self.M_plot = sparse.lil_matrix((n,  self.mx*self.my))
        self.M_plot[row_index,c1] = bx0*by0
        self.M_plot[row_index,c2] = bx1*by0
        self.M_plot[row_index,c3] = bx2*by0
        self.M_plot[row_index,c4] = bx0*by1
        self.M_plot[row_index,c5] = bx1*by1
        self.M_plot[row_index,c6] = bx2*by1
        self.M_plot[row_index,c7] = bx0*by2
        self.M_plot[row_index,c8] = bx1*by2
        self.M_plot[row_index,c9] = bx2*by2        

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

        bbx0 = (self.knot_space - taux_bar)/(self.knot_space) 
        bbx1 = (self.knot_space- taux_bar)/(2*self.knot_space)
        bbx2 = (taux_bar+self.knot_space)/(2*self.knot_space)
        bbx3 =  (taux_bar)/(self.knot_space)                   
        bbx4 = (2*self.knot_space- taux_bar)/(2*self.knot_space)
        bbx5 = (taux_bar)/(2*self.knot_space)

        bx0 = bbx0*bbx1
        bx1 = bbx0*bbx2 + bbx3*bbx4
        bx2 = bbx3*bbx5

        bby0 = (self.knot_space - tauy_bar)/(self.knot_space) 
        bby1 = (self.knot_space- tauy_bar)/(2*self.knot_space)
        bby2 = (tauy_bar+self.knot_space)/(2*self.knot_space)
        bby3 =  (tauy_bar)/(self.knot_space)                   
        bby4 = (2*self.knot_space- tauy_bar)/(2*self.knot_space)
        bby5 = (tauy_bar)/(2*self.knot_space)

        by0 = bby0*bby1
        by1 = bby0*bby2 + bby3*bby4
        by2 = bby3*bby5


        # Kronecker product
        c1 = (muy)*(self.mx)+(mux)
        c2 = (muy)*(self.mx)+(mux+1)
        c3 = (muy)*(self.mx)+(mux+2)

        c4 = (muy+1)*(self.mx)+(mux)
        c5 = (muy+1)*(self.mx)+(mux+1)
        c6 = (muy+1)*(self.mx)+(mux+2)

        c7 = (muy+2)*(self.mx)+(mux)
        c8 = (muy+2)*(self.mx)+(mux+1)
        c9 = (muy+2)*(self.mx)+(mux+2)       

        return bx0, bx1, bx2, by0, by1, by2, c1, c2, c3, c4, c5, c6, c7, c8, c9

    def update_spline_map(self, pts_occ, pts_free):
        tic = time.time()
        # Computing spline coefficients associated to occupied space
        # ############### Occupied space #########################
        n_occ = pts_occ.shape[1]
        bx0_occ, bx1_occ, bx2_occ, by0_occ, by1_occ, by2_occ, c1_occ, c2_occ, c3_occ, c4_occ, c5_occ, c6_occ, c7_occ, c8_occ, c9_occ = self.compute_spline(pts_occ)
        occ_cols = np.hstack((c1_occ,c2_occ,c3_occ,c4_occ, c5_occ,c6_occ,c7_occ,c8_occ, c9_occ))

        # ############### Free space ########################      
        n_free = pts_free.shape[1]
        bx0_free, bx1_free, bx2_free, by0_free, by1_free, by2_free, c1_free, c2_free, c3_free, c4_free, c5_free, c6_free, c7_free, c8_free, c9_free = self.compute_spline(pts_free)
        free_cols = np.hstack((c1_free,c2_free,c3_free,c4_free, c5_free,c6_free,c7_free,c8_free, c9_free))
        

        # Finding control points that have to be updated 
        cols_unique = np.hstack((occ_cols, free_cols))
        
        self.time[5] +=  time.time() - tic
        tic = time.time()
        B = np.zeros([n_free+n_occ,9])
        B[:,0] = np.concatenate([bx0_occ*by0_occ, bx0_free*by0_free])
        B[:,1] = np.concatenate([bx1_occ*by0_occ, bx1_free*by0_free])
        B[:,2] = np.concatenate([bx2_occ*by0_occ, bx2_free*by0_free])
        B[:,3] = np.concatenate([bx0_occ*by1_occ, bx0_free*by1_free])
        B[:,4] = np.concatenate([bx1_occ*by1_occ, bx1_free*by1_free])
        B[:,5] = np.concatenate([bx2_occ*by1_occ, bx2_free*by1_free])
        B[:,6] = np.concatenate([bx0_occ*by2_occ, bx0_free*by2_free])
        B[:,7] = np.concatenate([bx1_occ*by2_occ, bx1_free*by2_free])
        B[:,8] = np.concatenate([bx2_occ*by2_occ, bx2_free*by2_free])

        C = np.zeros([n_free+n_occ,9], dtype='int')
        C[:,0] = np.concatenate([c1_occ, c1_free])
        C[:,1] = np.concatenate([c2_occ, c2_free])        
        C[:,2] = np.concatenate([c3_occ, c3_free])
        C[:,3] = np.concatenate([c4_occ, c4_free])
        C[:,4] = np.concatenate([c5_occ, c5_free])        
        C[:,5] = np.concatenate([c6_occ, c6_free])
        C[:,6] = np.concatenate([c7_occ, c7_free])
        C[:,7] = np.concatenate([c8_occ, c8_free])        
        C[:,8] = np.concatenate([c9_occ, c9_free])

        # error
        e = np.concatenate([100*np.ones(n_occ), np.zeros(n_free)])
        e -= B[:,0]*self.ctrl_pts[C[:,0]]
        e -= B[:,1]*self.ctrl_pts[C[:,1]]
        e -= B[:,2]*self.ctrl_pts[C[:,2]]
        e -= B[:,3]*self.ctrl_pts[C[:,3]]
        e -= B[:,4]*self.ctrl_pts[C[:,4]]
        e -= B[:,5]*self.ctrl_pts[C[:,5]]
        e -= B[:,6]*self.ctrl_pts[C[:,6]]
        e -= B[:,7]*self.ctrl_pts[C[:,7]]        
        e -= B[:,8]*self.ctrl_pts[C[:,8]]                       

        cmin = np.min(cols_unique)
        cmax = np.max(cols_unique)
        gradient = np.zeros(cmax-cmin+1)
        gradient[C[:,0]-cmin] += B[:,0]*e
        gradient[C[:,1]-cmin] += B[:,1]*e
        gradient[C[:,2]-cmin] += B[:,2]*e
        gradient[C[:,3]-cmin] += B[:,3]*e
        gradient[C[:,4]-cmin] += B[:,4]*e
        gradient[C[:,5]-cmin] += B[:,5]*e
        gradient[C[:,6]-cmin] += B[:,6]*e
        gradient[C[:,7]-cmin] += B[:,7]*e        
        gradient[C[:,8]-cmin] += B[:,8]*e               
        
        self.ctrl_pts[cmin:cmax+1] += .01*gradient
        self.time[6] +=  time.time() - tic

        
    def compute_map_plot(self):
        return (self.M_plot@self.ctrl_pts).reshape(self.plot_size_x, self.plot_size_y)

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