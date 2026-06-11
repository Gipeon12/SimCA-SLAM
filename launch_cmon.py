import os
import csv
import time
import rclpy
import gtsam
import argparse
import threading
import traceback
import math as m
import numpy as np
import tkinter as tk
import scipy.ndimage as snd
from rclpy.node import Node
from tf2_ros import Buffer
from itertools import product
from tf2_msgs.msg import TFMessage
from collections import deque  # Double-ended queue
from simca_interface.msg import ObservationSet, ControlSet
from geometry_msgs.msg import TransformStamped
from rclpy.executors import MultiThreadedExecutor


# ============ Argument parser =========================================================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-cs", "--cellsize", type = float, default = 0.25, help = "Cell size in meters")
args = parser.parse_args()


# ================= Global parameters ==================================================================================================

CELL_SIZE = args.cellsize          # Size of the "low-resolution" cell in meters
SAMPLE_INC = 2                     # Sample incrementation: >1 to ignore some lidar points
XDIM_SIZE = YDIM_SIZE = 20.0       # Size of the 2D environment in meters
FRONT_DIST = 1.0                   # Distance (in meters) between known areas and the frontier line
SCALE = 50                         # Number of pixels per meter in the simulation
DELAY_MS = 400                     # Refresh rate in milliseconds
TAU_RDV = 100.0                    # Critical duration for allocating rendezvous tasks
XP_THRESH = 0.2                    # Exploration ratio threshold to start allocating tasks
FREE = False                       # Free space tracing for lasers at infinity (best if False)


# ============= Remarks ================================================================================================================

# In this file, we use the following naming conventions:
#   - (x, y) and its variants (as FLOATS) for coordinates expressed in METERS in the real world.
#   - (u, v) and its variants (as FLOATS) for coordinates expressed in PIXELS as on a canvas.
#   - (c, r) and its variants (as INTEGERS) for indices in a grid or an array.
#
#   NB:     (x, y)  [* SCALE]  ~>  (u,v)
#           (x, y)  [// CELL_SIZE > to INT]  ~>  (c, r)
#
# When creating the occupancy grid, we align the center of the central cell (odd number of cells per dimension) with the origin of the world frame.
# Since (x, y) is not always exactly at the center of a cell, we need to shift the scans coordinates to ensure that each point is assigned to the correct cell.
# The actual offset is given by the reduced coordinates of (x, y) within the surrounding cell: x_off = (x0 + x) % CELL_SIZE; y_off = (y0 + y) % CELL_SIZE.
#
# Formula to calculate yaw from quaternion:
#     yaw = atan2(2*(q4.q3+q1.q2), 1-2*(q2^2+q3^2)) [rad]
#
# Filling policy for the occupancy grid cells:
#   - Each cell needs to be populated with a certain amount of laser points to be considered full.
# We sketch a log-odds method for probabilistic occupancy grid (based on ISM: Inverse Sensor Model; Bayesian techniques and probabilistic inference).
# Forward model: generative description of the physics of the sensors, of the form p(z|m) where p is the probability of a measurement z given a map m.
#
# We define the occupancy log-odds for p = (0: certainly free) -> (1: certainly occupied).

def log_odds(p):
    return m.log(p/(1.0 - p))

def free_prob(l):
    # Return 1-p: probability of being "free"
    return 1 / (1 + m.exp(l))

LOGODDS = { "min":   log_odds(0.10),            # L_min  = -2.20
            "free":  log_odds(0.40),            # L_free = -0.41
            "void":  log_odds(0.50),            # L_void =  0.0
            "occ":   log_odds(0.70),            # L_occ  =  0.85
            "max":   log_odds(0.95), }          # L_max  =  2.94

# |L_occ| > |L_free| because hitting an obstacle is always stronger evidence than detecting nothing in a cell.
# Defining L_min and L_max allows the system to remain responsive and reversible. We want to avoid cells with overly confident status.


# ============== Utility functions =====================================================================================================

def as_color(p):
    q = int(p * 255)
    return f"#{q:02x}{q:02x}{q:02x}"
    
def get_namespaces():  # Nodes awareness
    enum = Node("mob_enumerator")
    # Allow DDS discovery to complete
    time.sleep(1.0)
    # Get mob names
    nodes = enum.get_node_names_and_namespaces()
    namespaces = []
    for node, ns in nodes:
        name = ns[1:]
        if ns != "/" and not name in namespaces:
            namespaces.append(name)
    enum.destroy_node()
    return sorted(namespaces)

def get_vlm_items(vlm_xs, vlm_ys):
    xs, ys = np.array(vlm_xs), np.array(vlm_ys)
    xb, yb = np.sum(xs)/len(xs), np.sum(ys)/len(ys)
    vtx_len = np.array([((x-xb)**2+(y-yb)**2)**0.5 for x, y in zip(xs, ys)])
    rho, phi = (xb**2+yb**2)**0.5, np.atan2(yb, xb)
    sign = np.roll(vtx_len, -np.argmax(vtx_len))
    return (rho, phi), sign

def match_landmarks(lm1_items, lm2_items, max_dv = 0.25, max_dp = 2.0):
    (sign1, x1, y1) = lm1_items
    (sign2, x2, y2) = lm2_items
    lm_dist = ((x2 - x1)**2 + (y2 - y1)**2)**0.5
    sgn_sqd = 0
    sign_cond = len(sign1) == len(sign2)
    prox_cond = lm_dist < max_dp
    if not sign_cond or not prox_cond:
        return False, None, lm_dist
    for v1, v2 in zip(sign1, sign2):
        if abs(v1 - v2) > max_dv:
            return False, None, lm_dist
        sgn_sqd += (v1 - v2)**2
    return True, sgn_sqd, lm_dist

def transform_frame(pose1, rge_b1, pose2, rge_b2):
    # Transformation of frame (R2) in frame (R1) using mutual observations
    [x1, y1, yaw1], (rho1, phi1) = pose1, rge_b1
    [x2, y2, yaw2], (rho2, phi2) = pose2, rge_b2
    # -----
    rho = (rho1 + rho2)/2    # Average of rho1 and rho2 (supposed to be equal anyway)
    r2 = (x2**2 + y2**2)**0.5
    psi2 = np.atan2(-y2, -x2)
    # -----
    Rot_12 = wrap(np.pi + (yaw1 - yaw2) + (phi1 - phi2))
    Rx_12 = float(x1 + rho * np.cos(yaw1 + phi1) + r2 * np.cos(Rot_12 + psi2))
    Ry_12 = float(y1 + rho * np.sin(yaw1 + phi1) + r2 * np.sin(Rot_12 + psi2))
    return Rx_12, Ry_12, Rot_12

def get_cartesian_sample(angles, ranges, yaw, range_limit, free_space = False, sample_increm = SAMPLE_INC):
    N = len(ranges)
    x_coords = []
    y_coords = []
    hit_flag = []
    for k in range(0, N, sample_increm):
        ang = float(angles[k]) + yaw
        rge = float(ranges[k])
        hit = True
        if rge == float('inf'):
            if not free_space: 
                continue
            else:
                rge = range_limit
                hit = False
        x_coords += [rge*m.cos(ang),]
        y_coords += [rge*m.sin(ang),]
        hit_flag += [hit,]
    return np.array(x_coords), np.array(y_coords), hit_flag

def push_on_cell(grid, c, r, hit = True, L_odds = LOGODDS):
    L = L_odds["occ"] if hit else L_odds["free"]
    l = grid[r, c] + L
    grid[r, c] = min(L_odds["max"], max(l, L_odds["min"]))

def ray_trace(grid, c0, r0, c1, r1, end_hit = True):
    # Trace light beam between center (c0, r0) and end point (c1, r1).
    # Indicate whether the light beam hit an occupied cell or not.
    n = max(abs(c1 - c0), abs(r1 - r0)) + 1
    ray_cs = np.linspace(c0, c1, n).astype(int)
    ray_rs = np.linspace(r0, r1, n).astype(int)
    ray_hits = [False] * (n-1) + [end_hit]
    for c, r, hit in zip(ray_cs, ray_rs, ray_hits):
        push_on_cell(grid, c, r, hit)

def get_lowres_grid(sc_angles, sc_ranges, yaw, range_limit, offset = (0, 0), free_space = FREE, cell_size = CELL_SIZE):
    range_resol = range_limit/cell_size
    nb_cells = round(2*range_resol + 1)
    c0 = r0 = nb_cells//2
    x0 = y0 = cell_size*nb_cells/2
    x_off, y_off = offset   # Offset due to the robot not being at the very center of the crossed cell
    grid = np.zeros((nb_cells, nb_cells))
    x_beams, y_beams, hit_flag = get_cartesian_sample(sc_angles, sc_ranges, yaw, range_limit, free_space)
    for k in range(len(hit_flag)):
        x_k, y_k, hitF_k = x_beams[k] + x_off, y_beams[k] + y_off, hit_flag[k]
        c_k, r_k = int((x0 + x_k)//cell_size), int((y0 - y_k)//cell_size)
        try:   # Ignore points that are marginally out of bounds
            ray_trace(grid, c0, r0, c_k, r_k, hitF_k)
        except:
            continue
    return grid

def covariance_ellipse(cov_xy, n_std=2.0):
    vals, vecs = np.linalg.eigh(cov_xy)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    width, height = 2 * n_std * np.sqrt(abs(vals))
    angle = m.atan2(vecs[1, 0], vecs[0, 0])
    return width, height, angle

def lm_goal_acceptance(rob_pose, lm_coords, rho_min=2.0, rho_max=8.0, fov=np.pi/2):
    # Reject landmarks that are outside the desired field of acceptance
    [x, y, yaw] = rob_pose
    [xL, yL] = lm_coords
    Dx, Dy = xL - x, yL - y
    rho = (Dx**2 + Dy**2)**0.5
    if rho < rho_min or rho > rho_max:
        return False
    phi = wrap(np.atan2(Dy, Dx) - yaw)
    if abs(phi) > fov/2:
        return False
    return True

def disk_kernel(frt_dist = FRONT_DIST, cell_size = CELL_SIZE):
    ncells = int(frt_dist/cell_size)
    ker_size = 2*ncells + 1
    ys, xs = np.ogrid[:ker_size,:ker_size]
    return (xs-(ker_size-1)/2)**2 + (ys-(ker_size-1)/2)**2 <= ncells**2

def sample_track(P0, P1, d0 = 1.):
    x0, y0 = P0
    x1, y1 = P1
    d = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
    n = round(d/d0) + 1
    track = np.linspace(P0, P1, n)
    return track, d

def distance_to_segment(segment, T):
    A, B = segment[0], segment[-1]
    a = ((T[0] - A[0])**2 + (T[1] - A[1])**2)**0.5
    b = ((T[0] - B[0])**2 + (T[1] - B[1])**2)**0.5
    c = ((B[0] - A[0])**2 + (B[1] - A[1])**2)**0.5
    p = (a + b + c)/2
    S = (p*(p-a)*(p-b)*(p-c))**0.5
    d = min(a, b) if max(a, b)**2 >= c**2 + min(a, b)**2 else 2*S/c
    return d


# =========== Wrap angles to [-pi, pi] =================================================================================================

wrap = lambda ang : (ang + np.pi)%(2*np.pi) - np.pi


# ============= Homogeneous transformation matrix (2D pose composition) and homogeneous pose vector ====================================

Htf = lambda x, y, yaw : np.array([[np.cos(yaw), -np.sin(yaw),  0,    x], 
                                   [np.sin(yaw),  np.cos(yaw),  0,    y],
                                   [0,            0,            1,  yaw],
                                   [0,            0,            0,    1]])

Uhp = lambda x, y, yaw : np.array([x, y, yaw, 1.])


# =========== Covariance matrix for VLM observations (polar coordinates) ===============================================================

# Cartesian covariance for one point
Sig_xy = lambda r, b, s1 = 0.05, s2 = 0.05 : np.array([[(np.cos(b)*s1)**2+(r*np.sin(b)*s2)**2, np.cos(b)*np.sin(b)*(s1**2-(r*s2)**2)],
                                                       [np.cos(b)*np.sin(b)*(s1**2-(r*s2)**2), (np.sin(b)*s1)**2+(r*np.cos(b)*s2)**2]])

# Jacobian to get back to polar coordinates (r = sqrt(x**2 + y**2), b = atan2(y, x))
Jac_rb = lambda x, y : np.array([[x/((x**2+y**2)**0.5), y/((x**2+y**2)**0.5)], [-y/(x**2+y**2), x/(x**2+y**2)]])

def get_vlm_polarcov(x_vals, y_vals):
    N = len(x_vals)
    xb, yb = 0, 0    # Barycenter
    Sig = np.zeros((2, 2))
    for x, y in zip(x_vals, y_vals):
        r, b = (x**2+y**2)**0.5, np.atan2(y, x)
        Sig += (1/N**2) * Sig_xy(r, b)
        xb += (1/N) * x
        yb += (1/N) * y
    J = Jac_rb(xb, yb)
    Sig_vlm = J @ Sig @ J.T
    return Sig_vlm


# =========== Covariance matrix for mutual observation (relative pose measurement) with c = cos() and s = sin() ========================

Sig_mut = lambda r, c, s, sig1 = 0.01, sig2 = 0.001 : np.array([[(c*sig1)**2+(r*s*sig2)**2,    c*s*(sig1**2-r*sig2**2),   -r*s*sig2**2],
                                                                [ c*s*(sig1**2-r*sig2**2),    (s*sig1)**2+(r*c*sig2)**2,   r*c*sig2**2],
                                                                [-r*s*sig2**2,                 r*c*sig2**2,                  2*sig2**2]])


# ============== Boolean to determine whether or not two inter-robot observations are mutual ===========================================

is_mut = lambda a, b : a['source'] == b['target'] and a['target'] == b['source']


# ============= Observation Listener for data gathering (buffering) ====================================================================

class DataGatherer:

    def __init__(self, group_size = 1):
        # Optimization graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        params = gtsam.ISAM2Params()
        params.setFactorization("QR")
        self.isam = gtsam.ISAM2(params)
        # Data recording
        self.time = 0.
        self.nrobs = group_size
        self.seen_keys = set()    # All keys that will be used during runtime
        self.idx_count = {"lm":1} # Idx counter to create a new key
        self.id_symbol = {}       # id_symbol[rob_i] = '\x01', '\x02', '\x03' or other depending on the number of robots (character as unique identifier)
        self.data_hist = {}       # data_hist[rob_i] = [data1_i, data2_i, ...]  |  data_i = (gtsam_key, sim_time, pose, sc_angs, sc_rges, rge_lim)
        self.traj_buff = {}       # traj_buff[rob_i] = deque(maxlen = 100) of last (x_i, y_i, yaw_i)
        self.stat_eval = {}       # stat_eval[rob_i] = (status: "free_roaming", "busy" or "idle"; tau_i: duration since last rendezvous)
        self.landmarks = {}       # landmarks[key] = (sign, x, y, weight, key of last visitor)
        # World -> Odom transformations
        self.w2od_tf = {}         # w2od_tf[rob_i] = [x_od, y_od, yaw_od] at the last update
        # Inter-robots relations
        self.mut_obsrv = []       # List (buffer) of all mutual observations (described as dictionaries)
        self.global_tf = {}       # global_tf[rob_i] = (x0_i, y0_i, yaw0_i) in global frame (all zeros while robots don't get mutual observations)
        self.relatv_tf = {}       # relatv_tf[rob_i] = {rob_j: (x_ij, y_ij, yaw_ij), ...} -> relative transformations between agents (rob_j in rob_i frame, j != i)
        self.rob_to_link = set()  # Names of the non-leader robots (waiting to be connected to the leader)
        self.allRelated = False if self.nrobs > 1 else True   # Becomes True when all robots have been connected to the leader
        # Thread management
        self.lock = threading.Lock()

    def optimize_graph(self):
        self.isam.update(self.graph, self.initial)
        self.graph.resize(0)
        self.initial.clear()
        self.result = self.isam.calculateEstimate()
        self.marginals = gtsam.Marginals(self.isam.getFactorsUnsafe(), self.result)

    def set_role(self, rob_ns, isLeader = False):
        if isLeader:
            self.lead_ns = rob_ns
        else:
            self.rob_to_link.add(rob_ns)
        self.relatv_tf[rob_ns] = {}
    
    def update_stat_eval(self, rob_ns, work_status):
        # Also increment time since last contact with another agent
        t0 = self.data_hist[rob_ns][-2][1]
        t1 = self.data_hist[rob_ns][-1][1]
        tau = self.stat_eval[rob_ns][1]
        self.stat_eval[rob_ns] = (work_status, tau + t1 - t0)
    
    def factor_global_tf(self):
        for ns, (x, y, yaw) in self.relatv_tf[self.lead_ns].items():
            self.global_tf[ns] = x, y, yaw
        # Add prior for the leader (whose initial pose serves as origin of the world frame)
        key0 = self.data_hist[self.lead_ns][0][0]
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-6, 1e-6, 1e-6])
        self.graph.add(gtsam.PriorFactorPose2(key0, gtsam.Pose2(0., 0., 0.), prior_noise))
    
    def recover_initial_guess(self):
        for rob_ns in self.data_hist:
            x0, y0, yaw0 = self.global_tf[rob_ns]
            for (key, _, [x, y, yaw], _, _, _) in self.data_hist[rob_ns]:
                tf_pose = Htf(x0, y0, yaw0) @ Uhp(x, y, yaw)
                pose = gtsam.Pose2(tf_pose[0], tf_pose[1], tf_pose[2])
                self.initial.insert(key, pose)
    
    def factor_pose(self, rob_ns):
        (key0, _, [x0, y0, yaw0], _, _, _) = self.data_hist[rob_ns][-2]
        (key1, _, [x1, y1, yaw1], _, _, _) = self.data_hist[rob_ns][-1]
        # Relative transformation (pose1 in pose0 frame)
        c, s = np.cos(yaw0), np.sin(yaw0)
        Dx =  c*(x1 - x0) + s*(y1 - y0)
        Dy = -s*(x1 - x0) + c*(y1 - y0)
        Dyaw = wrap(yaw1 - yaw0)    # Wrap to [-pi, pi] to comply with GTSAM node convention
        rtf_pose = gtsam.Pose2(Dx, Dy, Dyaw)
        # TODO: Covariance propagation from relative motion
        # Covariance policy: diagonal matrix with small coefficients (assumption of strong underlying SLAM)
        rtf_noise = gtsam.noiseModel.Diagonal.Sigmas([1e-4, 1e-4, 5e-5])    # Note: creates matrix R such that R x R.T = Sigma
        # Look for ill-conditioned systems with the condition number (the smaller the better) -> print(np.linalg.cond(Sigma))
        self.graph.add(gtsam.BetweenFactorPose2(key0, key1, rtf_pose, rtf_noise))
        if self.allRelated and (key1 not in self.seen_keys):
            self.initial.insert(key1, gtsam.Pose2(x1, y1, yaw1))
            self.seen_keys.add(key1)

    def factor_vlm(self, rob_ns, vlm_obs, excl_zone = 1.0):
        # Unpacking data
        (key_rob, _, [x, y, yaw], _, _, _) = self.data_hist[rob_ns][-1]
        [vlm_xs, vlm_ys] = vlm_obs
        (rho, phi), sign1 = get_vlm_items(vlm_xs, vlm_ys)
        # Initialize items
        key_vlm, w1 = None, 1
        x_lm1, y_lm1 = x + rho*np.cos(yaw + phi), y + rho*np.sin(yaw + phi)
        sqd_ref = float('inf')
        lm_dist = [float('inf')]
        found_match = False
        for key2, (sign2, x_lm2, y_lm2, w2, last_key_rob) in self.landmarks.items():
            lm_match, sgn_sqd, lm_d = match_landmarks(lm1_items = (sign1, x_lm1, y_lm1), lm2_items = (sign2, x_lm2, y_lm2))
            lm_dist.append(lm_d)
            if lm_match:
                found_match = True
            else:
                continue
            if sgn_sqd < sqd_ref:    # Register best match among all known landmarks
                key_vlm, sign0, x_lm0, y_lm0, w0 = key2, sign2, x_lm2, y_lm2, w2
                sqd_ref = sgn_sqd
        if not found_match and min(lm_dist) < excl_zone:    # If another existing landmark is close enough (without matching), we don't clutter the area by adding a new one
            return
        elif key_vlm == None:
            key_vlm = gtsam.symbol('z', self.idx_count["lm"])
            self.idx_count["lm"] += 1
        else:    # Averaging coordinates and signature for refinement
            w1 = w0 + 1
            sign1 = (w0 * sign0 + sign1)/w1
            x_lm1 = (w0 * x_lm0 + x_lm1)/w1
            y_lm1 = (w0 * y_lm0 + y_lm1)/w1
        # Update record and add a node to the graph
        self.landmarks[key_vlm] = (sign1, x_lm1, y_lm1, w1, key_rob)
        Sig_vlm = get_vlm_polarcov(vlm_xs, vlm_ys)
        obs_noise = gtsam.noiseModel.Gaussian.Covariance(Sig_vlm)
        self.graph.add(gtsam.BearingRangeFactor2D(key_rob, key_vlm, gtsam.Rot2(phi), rho, obs_noise))
        # Set initial guess for the landmark location (in global frame) if not already in the list
        if key_vlm not in self.seen_keys:
            self.initial.insert(key_vlm, gtsam.Point2(x_lm1, y_lm1))
            self.seen_keys.add(key_vlm)
        # Trigger graph optimization after re-observing a landmark
        #if w1 > 1 :
        #    self.optimize_graph()

    def factor_mutobs(self, obs_dct1):
        for obs_dct2 in self.mut_obsrv:
            if obs_dct1['time'] == obs_dct2['time'] and is_mut(obs_dct1, obs_dct2):
                key1, (rho1, phi1) = obs_dct1['key'], obs_dct1['rge_bear']
                key2, (rho2, phi2) = obs_dct2['key'], obs_dct2['rge_bear']
                # Rob2 factor pose in Rob1 body frame (cartesian coordinates)
                x12, y12, yaw12 = rho1*np.cos(phi1), rho1*np.sin(phi1), wrap(np.pi + phi1 - phi2)
                pose12 = gtsam.Pose2(x12, y12, yaw12)
                Sig12 = Sig_mut(rho1, np.cos(phi1), np.sin(phi1))
                noise12 = gtsam.noiseModel.Gaussian.Covariance(Sig12)
                self.graph.add(gtsam.BetweenFactorPose2(key1, key2, pose12, noise12))
                # Reinitialize duration since last contact
                rob_ns = obs_dct1['source']
                self.stat_eval[rob_ns] = (self.stat_eval[rob_ns][0], 0.)
                # Find transformation in leader frame
                if not self.allRelated:
                    self.cast_transform(obs_dct1, obs_dct2)
        self.mut_obsrv.append(obs_dct1)
    
    def cast_transform(self, obs_dct1, obs_dct2):
        rob1, pose1, rge_b1 = obs_dct1['source'], obs_dct1['pose'], obs_dct1['rge_bear']
        rob2, pose2, rge_b2 = obs_dct2['source'], obs_dct2['pose'], obs_dct2['rge_bear']
        # Mutual transformations
        if rob2 not in self.relatv_tf[rob1]:
            self.relatv_tf[rob1][rob2] = transform_frame(pose1, rge_b1, pose2, rge_b2)
            self.relatv_tf[rob2][rob1] = transform_frame(pose2, rge_b2, pose1, rge_b1)
        # Create remaining transformations if possible (indirect connections to leader)
        for rob_i, (Rx_i, Ry_i, Rot_i) in list(self.relatv_tf[self.lead_ns].items()):   # Fix the list to allow mutations during iteration
            for rob_j, (Rx_ij, Ry_ij, Rot_ij) in list(self.relatv_tf[rob_i].items()):
                if rob_j not in self.relatv_tf[self.lead_ns]:
                    htf_j = Htf(Rx_i, Ry_i, Rot_i) @ Uhp(Rx_ij, Ry_ij, Rot_ij)
                    self.relatv_tf[self.lead_ns][rob_j] = tuple(htf_j[:3])
        # If all robots have been connected to the leader, we can define a global frame
        if self.rob_to_link.issubset(set(self.relatv_tf[self.lead_ns].keys())):
            self.allRelated = True
            self.factor_global_tf()
            self.recover_initial_guess()

    def get_last_data(self, rob_ns):
        (_, _, rob_pose, sc_angles, sc_ranges, rge_lim) = self.data_hist[rob_ns][-1]
        return rob_pose, sc_angles, sc_ranges, rge_lim  
    
    def get_traj_data(self, rob_ns):
        return list(self.traj_buff[rob_ns])
    
    def get_w2od_tf(self, rob_ns):
        return self.w2od_tf[rob_ns]
    
    def get_stat_eval(self, rob_ns):
        return self.stat_eval[rob_ns]
    
    def get_landmarks_items(self):
        lm_items = []
        for key in self.landmarks:
            if hasattr(self, "result") and self.result.exists(key):
                lm_p = self.result.atPoint2(key)
                lm_cov = self.marginals.marginalCovariance(key)
                lm_items.append((lm_p[0], lm_p[1], lm_cov))
            else:
                _, x_lm, y_lm, w, _ = self.landmarks[key]
                lm_items.append((key, x_lm, y_lm, w, 5e-3*np.eye(2))) 
        return lm_items
    
    def get_time(self):
        return self.time
    
    def process(self, rob_ns, sim_time, slam_pose, sc_angles, sc_ranges, rge_lim, work_status, mut_obs, vlm_obs, u_odom):
        with self.lock:
            if rob_ns not in self.id_symbol:
                self.id_symbol[rob_ns] = chr(len(self.id_symbol) + 1)
                self.idx_count[rob_ns] = 1
                self.global_tf[rob_ns] = 0., 0., 0.
                self.data_hist[rob_ns] = []
                self.traj_buff[rob_ns] = deque(maxlen = 100)
                self.stat_eval[rob_ns] = ("free_roaming", 0.)
            # Create a new pose key
            key = gtsam.symbol(self.id_symbol[rob_ns], self.idx_count[rob_ns])
            self.idx_count[rob_ns] += 1
            # Transform to global coordinates
            x0, y0, yaw0 = self.global_tf[rob_ns]
            htf_pose = Htf(x0, y0, yaw0) @ slam_pose
            [x, y, yaw] = htf_pose[:3]
            self.data_hist[rob_ns].append((key, sim_time, [x, y, yaw], sc_angles, sc_ranges, rge_lim))
            self.traj_buff[rob_ns].append((x, y, yaw))
            # Update World -> Odom transform
            self.w2od_tf[rob_ns] = Htf(x0, y0, yaw0) @ u_odom
            if len(self.data_hist[rob_ns]) > 1:
                self.update_stat_eval(rob_ns, work_status)
                self.factor_pose(rob_ns)
            if len(mut_obs.items()):    # Priority given to mutual observations (filter false positives for the VLM)
                for targ_ns, rge_b in mut_obs.items():
                    obs_dct = {'time':round(sim_time, 1), 'source':rob_ns, 'target':targ_ns, 'rge_bear':rge_b, 'pose':[x, y, yaw], 'key':key}
                    self.factor_mutobs(obs_dct)
            elif len(vlm_obs[0]) > 1 and self.allRelated:    # Wait for the global frame to be defined to introduce the VLM
                self.factor_vlm(rob_ns, vlm_obs)
            # Update time
            self.time = sim_time if sim_time > self.time else self.time
            #if self.allRelated:
            #    self.optimize_graph()

class ObservationListener(Node):

    def __init__(self, data_gatherer, rob_ns = 'rob1', isLeader = False, step_sec = DELAY_MS/1000):
        super().__init__(f'{rob_ns}_observation_listener')
        self._Nspace = rob_ns
        self._DataGath = data_gatherer
        self._DataGath.set_role(rob_ns, isLeader)
        self._TFbuf = Buffer()
        self._OBSsub = self.create_subscription(ObservationSet, f"{rob_ns}/observation", self.obs_callback, 10)
        self._TFsub = self.create_subscription(TFMessage, f"{rob_ns}/tf", self.tf_callback, 10)
        self.get_logger().info(f'Subscribed to /{rob_ns}/observation and /{rob_ns}/tf')
        self._timer = self.create_timer(step_sec, self.cast_observation)

    def obs_callback(self, msg: ObservationSet):
        self.sim_time = msg.sim_time
        self.odom_pose = msg.odom_pose
        self.scan_angs = msg.scan_angles
        self.scan_rges = msg.scan_ranges
        self.rge_limit = msg.laser_limit
        self.work_status = msg.status
        self.mut_obs = {rob_name: (rho, phi) for rob_name, rho, phi in zip(msg.rob_names, msg.obs_ranges, msg.obs_angles)}
        self.vlm_obs = [msg.vlm_xs, msg.vlm_ys]
    
    def tf_callback(self, msg: TFMessage):
        for t in msg.transforms:
            self._TFbuf.set_transform(t, 'default_authority')
        
    def cast_observation(self):
        try:
            [x, y, yaw] = self.odom_pose
            tf_od = self._TFbuf.lookup_transform('map', 'odom', rclpy.time.Time())
            x_od, y_od, q = tf_od.transform.translation.x, tf_od.transform.translation.y, tf_od.transform.rotation
            yaw_od = np.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            # Pose composition from odometry and latest SLAM correction
            self.slam_pose = Htf(x_od, y_od, yaw_od) @ Uhp(x, y, yaw)
            u_odom = Uhp(x_od, y_od, yaw_od)
            self._DataGath.process(self._Nspace, self.sim_time, self.slam_pose, self.scan_angs, self.scan_rges, self.rge_limit, self.work_status, self.mut_obs, self.vlm_obs, u_odom)
        except AttributeError:
            pass
        except Exception as e:
            self.get_logger().warn(f"{e}")
            traceback.print_exc()
            

# ================= Control Publisher ===============

class GlobalControlPublisher(Node):

    def __init__(self):
        super().__init__('global_control_publisher')
        self.ctrl_publishers = {}
        self.lock = threading.Lock()
    
    def add_agent(self, rob_ns):
        with self.lock:
            self.ctrl_publishers[rob_ns] = self.create_publisher(ControlSet, f"{rob_ns}/control", 10)
    
    def create_message(self, w2od_tf, waypts):
        msg = ControlSet()
        msg.w2od_tf = w2od_tf[:3]
        msg.wpts_x = [w[0] for w in waypts]
        msg.wpts_y = [w[1] for w in waypts]
        return msg
    
    def send_control(self, rob_ns, w2od_tf, waypts):
        msg = self.create_message(w2od_tf, waypts)
        with self.lock:
            self.ctrl_publishers[rob_ns].publish(msg)


# ============== Virtual Map Objects ================

class LandmarkCell:

    def __init__(self, r, c):
        self.r = r
        self.c = c
        self.covariance = np.eye(2)
        self.logodds = 0.0
    
    def push_data(self, log_increment = None, cov_matrix = None, L_min = LOGODDS["min"], L_max = LOGODDS["max"]):
        if log_increment is not None:
            l = self.logodds + log_increment
            self.logodds = min(L_max, max(l, L_min))
        if cov_matrix is not None:
            # TODO: update the covariance accordingly
            self.covariance = cov_matrix

class OccupancyGrid:

    def __init__(self, cell_size = CELL_SIZE, xdim_size = XDIM_SIZE, ydim_size = YDIM_SIZE):
        self.Csize = cell_size
        self.Xsize = xdim_size
        self.Ysize = ydim_size
        self.Nrows = m.ceil(ydim_size/cell_size)
        self.Ncols = m.ceil(xdim_size/cell_size)
        self.cells = [[LandmarkCell(r, c) for c in range(self.Ncols)] for r in range(self.Nrows)]
        self.nborKernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        
    def local_update(self, cG, rG, local_grid, cov_xy):
        nrows, ncols = np.shape(local_grid)
        for (i, j), l in np.ndenumerate(local_grid):
            r, c = rG - nrows//2 + i, cG - ncols//2 + j
            if min(c, r, self.Ncols-1-c, self.Nrows-1-r) < 0:  # Ignore values outside the map
                continue
            else:
                self.cells[r][c].push_data(log_increment = l, cov_matrix = cov_xy)
        
    def get_prob_grid(self):
        grid = np.full((self.Nrows, self.Ncols), 0.5)
        for row in self.cells:
            for lm_cell in row:
                grid[lm_cell.r, lm_cell.c] = free_prob(lm_cell.logodds)
        return grid
    
    def locate_reachable_frontiers(self, free_thresh = 0.4, occup_thresh = 0.7):
        prob_map = 1 - self.get_prob_grid()
        free_cells = prob_map < free_thresh
        occup_cells = prob_map > occup_thresh
        # Known cells are all cells already identified as free or occupied
        known_cells = free_cells | occup_cells
        unknown_cells = np.logical_not(known_cells)
        # Calculate the exploration ratio
        xp_rat = np.mean(known_cells)
        # Calculate the number of free cells in the surroundings of each cell
        free_nbors = snd.convolve(free_cells.astype(int), self.nborKernel, mode='constant', cval=0)    
        # Calculate the number of occupied cells in the surroundings of each cell
        occup_nbors = snd.convolve(occup_cells.astype(int), self.nborKernel, mode='constant', cval=0)
        # Reachable frontier cells are unknown cells with free surroundings
        frontiers = unknown_cells & (free_nbors > 0) & (free_nbors < 3) & (occup_nbors == 0)
        frontiers[:10, :] = frontiers[-10:, :] = False    # Excluding margin cells
        frontiers[:, :10] = frontiers[:, -10:] = False
        front_idx = np.argwhere(frontiers)
        return front_idx, xp_rat, known_cells


# ================ Centralized Monitoring System ==============

class CentralMonitor:

    def __init__(self, data_gatherer, controller, scale = SCALE, delay = DELAY_MS, tau_rdv = TAU_RDV, xp_thresh = XP_THRESH):
        self.marker = {1:"red", 2:"green", 3:"blue", 4:"orange", 5:"yellow", 6:"purple"}
        self.gatherer = data_gatherer
        self.controller = controller
        self.scale = scale
        self.delay = delay
        self.root = tk.Tk()
        self.root.title("Centralized Monitoring System")
        self.OGrid = OccupancyGrid()
        self.ncols = self.OGrid.Ncols
        self.nrows = self.OGrid.Nrows
        self.csize = self.OGrid.Csize
        self.x0 = self.ncols * self.csize / 2
        self.y0 = self.nrows * self.csize / 2
        wd_width = self.ncols * self.csize * self.scale
        wd_height = self.nrows * self.csize * self.scale
        self.u0 = wd_width/2
        self.v0 = wd_height/2
        self.grid_cvs = tk.Canvas(self.root, width=wd_width, height=wd_height)
        self.grid_cvs.pack(fill=tk.BOTH, expand=True)
        self.cell_cvs = [[None]*self.ncols for _ in range(self.nrows)]
        self.traj_cvs = {}
        self.head_cvs = {}
        self.elps_cvs = {}
        self.init_canvas()
        self.data_grid = np.full((self.nrows, self.ncols), 0.5)
        # Characteristic values for task allocation
        self.Nr = self.gatherer.nrobs                           # Number of robots
        W, H = self.ncols*self.csize, self.nrows*self.csize     # Height and Width of the map
        self.sig_d = (W*H/(4*self.Nr))**0.5                     # Characteristic distance for travel cost
        self.tau_c = tau_rdv                                    # Critical duration for seeking rendezvous
        self.xp_thresh = xp_thresh
        # Data recording
        self._last = {"time":-1.}
        self._pathLen = {}
        self.indicators = []

    def init_canvas(self):
        npxl = self.csize * self.scale
        for r in range(self.nrows):
            for c in range(self.ncols):
                self.cell_cvs[r][c] = self.grid_cvs.create_rectangle(c*npxl, r*npxl, (c+1)*npxl, (r+1)*npxl, fill=as_color(0.5), outline="gray")

    def ground_to_canvas(self, x, y):
        return self.u0 + x * self.scale, self.v0 - y * self.scale
    
    def ground_to_grid(self, x, y):
        return int((self.x0 + x) // self.csize), int((self.y0 - y) // self.csize)
        
    def grid_to_canvas(self, r, c):
        return (c + 0.5) * self.csize * self.scale, (r + 0.5) * self.csize * self.scale
    
    def grid_to_ground(self, r, c):
        return (c + 0.5) * self.csize - self.x0, self.y0 - (r + 0.5) * self.csize
        
    def cell_offset(self, x, y):
        x_off, y_off = (self.x0 + x) % self.csize, (self.y0 + y) % self.csize
        return (x_off, y_off)
    
    def refresh_grid(self):
        latest_grid = self.OGrid.get_prob_grid()
        differ = self.data_grid != latest_grid  # Find cells that changed
        for r, c in zip(*np.where(differ)):
            p = latest_grid[r, c]
            self.data_grid[r, c] = p
            self.grid_cvs.itemconfig(self.cell_cvs[r][c], fill=as_color(p))
    
    def draw_trajectory(self, rob_ns, traj_data):
        pts = []
        for (x, y, yaw) in traj_data:
            u, v = self.ground_to_canvas(x, y)
            pts.extend([u, v])
        pts = pts if len(pts) >= 4 else [0]*4
        self.grid_cvs.coords(self.traj_cvs[rob_ns], *pts)
    
    def draw_head_arrow(self, rob_ns, x, y, yaw, r = 0.5):
        uT, vT = self.ground_to_canvas(x, y)
        uH, vH = self.ground_to_canvas(x + r * m.cos(yaw), y + r * m.sin(yaw))
        self.grid_cvs.coords(self.head_cvs[rob_ns], uT, vT, uH, vH)

    def draw_marker(self, u, v, col, tag, size = 6, wid = 2):
        self.grid_cvs.create_line(u-size, v,      u+size, v,      fill=col, width=wid, tags=tag)
        self.grid_cvs.create_line(u,      v-size, u,      v+size, fill=col, width=wid, tags=tag)
    
    def draw_lm_covariance(self, x, y, cov_xy, col, tag):
        w, h, theta = covariance_ellipse(cov_xy, n_std = 10)
        pts = []
        for ang in np.linspace(0, 2 * m.pi, 40):
            ex = (w / 2) * m.cos(ang)
            ey = (h / 2) * m.sin(ang)
            rx = ex * m.cos(theta) - ey * m.sin(theta)
            ry = ex * m.sin(theta) + ey * m.cos(theta)
            u, v = self.ground_to_canvas(x + rx, y + ry)
            pts.extend([u, v])
        self.grid_cvs.create_polygon(pts, outline=col, fill="", width=2, tags=tag)
    
    def draw_landmarks(self, lm_items, col = "magenta", tag = "lm"):
        self.grid_cvs.delete(tag)
        for (_, x, y, w, cov_xy) in lm_items:
            u, v = self.ground_to_canvas(x, y)
            self.draw_marker(u, v, col, tag, size = 8, wid = 4)
            self.grid_cvs.create_text(u + 8, v - 12, text=str(w), font=("Arial", 10), fill=col, tags=tag)
            #self.draw_lm_covariance(x, y, cov_xy, col, tag)
    
    def draw_frontiers(self, front_idx, tag = "frt"):
        self.grid_cvs.delete(tag)
        for [r, c] in front_idx:
            u, v = self.grid_to_canvas(r, c)
            self.draw_marker(u, v, col = "cyan", tag = tag)

    def draw_pose_covariance(self, rob_ns, x, y, cov_xy):
        w, h, theta = covariance_ellipse(cov_xy)
        pts = []
        for ang in np.linspace(0, 2 * m.pi, 40):
            ex = (w / 2) * m.cos(ang)
            ey = (h / 2) * m.sin(ang)
            rx = ex * m.cos(theta) - ey * m.sin(theta)
            ry = ex * m.sin(theta) + ey * m.cos(theta)
            u, v = self.ground_to_canvas(x + rx, y + ry)
            pts.extend([u, v])
        self.grid_cvs.coords(self.elps_cvs[rob_ns], *pts)
    
    def update_global_state_factors(self, xp_rat, lm_items, indiv_states, lamb0 = 3., lamb1 = 1., lamb2 = 0.5):
        tau_robs = [tau for _, (_, _, _, tau) in indiv_states.items()]
        w_lms = [np.exp(1-w) for (_, _, _, w, _) in lm_items]
        self.L_exp = lamb0 * np.exp(-xp_rat**2)
        self.L_vlm = lamb1 * np.log(1 + sum(w_lms))
        self.L_rdv = lamb2 * np.log(1 + sum(tau_robs)/(self.Nr*self.tau_c))

    def generate_individual_goals(self, rob_ns, rob_pose, lm_items, front_idx, xp_rat):
        rob_goals = []
        [x, y, yaw] = rob_pose
        if xp_rat < self.xp_thresh or len(front_idx) + len(lm_items) == 0:
            return rob_goals
        c, r = self.ground_to_grid(x, y)
        rob_idx = np.array([r, c])
        # 1: Find nearest frontier to current position
        if len(front_idx) > 0:
            front_dist = np.linalg.norm(front_idx - rob_idx, axis=1)
            i0 = np.argmin(front_dist)
            [rF0, cF0] = front_idx[i0]
            xF, yF = self.grid_to_ground(rF0, cF0)
            uF, vF = self.grid_to_canvas(rF0, cF0)
            self.draw_marker(uF, vF, col = "yellow", tag = 'frt')    # Show potential target
            rob_goals.append((rob_ns, "front", (xF, yF)))
        # 2: Find potential landmark revisitations
        for (_, xL, yL, _, _) in lm_items:
            if lm_goal_acceptance(rob_pose, [xL, yL]):
                rob_goals.append((rob_ns, "vlm", (xL, yL)))
        return rob_goals
    
    def generate_rdv_goals(self, free_robs, xp_rat, max_lookup_dist = 5.):
        rdv_goals, pairs = [], []
        if xp_rat < self.xp_thresh:
            return rdv_goals
        for rob1, p1 in free_robs:
            for rob2, p2 in free_robs:
                if rob2 == rob1 or (rob2, rob1) in pairs:
                    continue
                xR, yR = (p1[0] + p2[0])/2, (p1[1] + p2[1])/2
                d = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                if d < max_lookup_dist:
                    rdv_goals.append((rob1, rob2, (xR, yR)))
                    pairs.append((rob1, rob2))
        return rdv_goals
    
    def group_goals(self, indiv_goals, rdv_goals):
        if len(indiv_goals) == 0:
            return []
        if len(indiv_goals) == 1:
            return [(goal) for goal in indiv_goals]
        group_goals = list(product(*indiv_goals))
        if len(rdv_goals) == 1:
            rob1, rob2, rdv_target = rdv_goals[0]
            group_goals.append(((rob1, "rdv", rdv_target), (rob2, "rdv", rdv_target)))
            return group_goals
        for rob_goals in indiv_goals:
            for rob1, task, target in rob_goals:
                for rdv in rdv_goals:
                    if rob1 in rdv:
                        continue
                    rob2, rob3, rdv_target = rdv
                    group_goals.append(((rob1, task, target), (rob2, "rdv", rdv_target), (rob3, "rdv", rdv_target)))
        return group_goals
    
    def exploration_gain(self, track, known_cells, kern = disk_kernel(frt_dist = 1.6), k_exp = 100.):
        mask = np.copy(known_cells)
        (nR, nC) = np.shape(kern)
        for [x, y] in track:
            c, r = self.ground_to_grid(x, y)
            r0, r1 = r-nR//2, r+nR-nR//2
            c0, c1 = c-nC//2, c+nC-nC//2
            loc = mask[r0:r1, c0:c1]
            new = loc | kern
            mask[r0:r1, c0:c1] = new
        new_cells = np.logical_xor(known_cells, mask)
        g_exp = k_exp * np.mean(new_cells)
        return g_exp, mask

    def visitation_gain(self, track, lm_items, s_prox = 0.5, k_vlm = 1.):
        g_vlm = 0.
        for (_, x, y, w, _) in lm_items:
            d = distance_to_segment(track, [x, y])
            g_vlm += k_vlm * np.exp(-(d/s_prox)**2 + 1 - w)
        return g_vlm

    def rendezvous_gain(self, tau, task, k_rdv = 1.):
        g_rdv = k_rdv * (1 - np.exp(-tau/self.tau_c)) if task == "rdv" else 0.
        return g_rdv

    def goals_utility(self, group_goals, indiv_states, known_cells, lm_items):
        U_goals = []
        for goals in group_goals:
            u_goals = 0.
            known_mask = np.copy(known_cells)    # We evaluate the virtual expansion of the map for each combination of goals
            for (rob, task, (xG, yG)) in goals:
                [x, y, yaw], _, _, tau = indiv_states[rob]
                track, d_travel = sample_track((x, y), (xG, yG))
                c_task = np.exp(-(d_travel/self.sig_d)**2)    # Cost factor depending on the travel distance
                g_exp, known_mask = self.exploration_gain(track, known_mask)
                g_vlm = self.visitation_gain(track, lm_items)
                g_rdv = self.rendezvous_gain(tau, task)
                u_goals += c_task * (self.L_exp * g_exp + self.L_vlm * g_vlm + self.L_rdv * g_rdv)
            U_goals.append(u_goals)
        return U_goals
    
    def frontier_only_goals(self, free_robs, indiv_states, lm_items, front_idx, xp_rat):
        chosen_goals = []
        for rob_ns, rob_pose in free_robs:
            rob_goals = self.generate_individual_goals(rob_ns, rob_pose, lm_items, front_idx, xp_rat)
            for g in rob_goals:
                if g[1] == "front":
                    chosen_goals.append(g)
                    break
        return chosen_goals

    def all_tasks_goals(self, free_robs, indiv_states, lm_items, front_idx, xp_rat, known_cells):
        indiv_goals = []
        for rob_ns, rob_pose in free_robs:
            rob_goals = self.generate_individual_goals(rob_ns, rob_pose, lm_items, front_idx, xp_rat)
            if len(rob_goals) > 0: 
                indiv_goals.append(rob_goals)
        chosen_goals = []
        rdv_goals = self.generate_rdv_goals(free_robs, xp_rat)
        group_goals = self.group_goals(indiv_goals, rdv_goals)
        U_goals = self.goals_utility(group_goals, indiv_states, known_cells, lm_items)
        if len(U_goals) > 0:    # Determining the set of goals with best expected gain
            i_max = np.argmax(U_goals)
            chosen_goals = [g for g in group_goals[i_max]]
        return chosen_goals

    def update(self):
        i = 0
        with self.gatherer.lock:
            indiv_states = {}
            for rob_ns in self.gatherer.data_hist:
                if rob_ns not in self.traj_cvs:
                    try:
                        i = int(rob_ns[-1])
                    except:
                        i += 1
                    self.traj_cvs[rob_ns] = self.grid_cvs.create_line(0, 0, 0, 0, fill=self.marker[i], dash=(2, 2))
                    self.head_cvs[rob_ns] = self.grid_cvs.create_line(0, 0, 0, 0, arrow=tk.LAST, width=2, fill=self.marker[i])
                    self.elps_cvs[rob_ns] = self.grid_cvs.create_polygon(0, 0, 0, 0, outline=self.marker[i], fill="", width=2)
                [x, y, yaw], sc_angles, sc_ranges, rge_lim = self.gatherer.get_last_data(rob_ns)
                w2od_tf = self.gatherer.get_w2od_tf(rob_ns)
                work_status, tau_stale = self.gatherer.get_stat_eval(rob_ns)
                indiv_states[rob_ns] = [x, y, yaw], w2od_tf, work_status, tau_stale
                xy_offset = self.cell_offset(x, y)
                sub_grid = get_lowres_grid(sc_angles, sc_ranges, yaw, rge_lim, xy_offset)
                cG, rG = self.ground_to_grid(x, y)
                # Update grid
                self.OGrid.local_update(cG, rG, sub_grid, cov_xy = None)
                self.refresh_grid()
                # Draw trajectory data
                traj_data = self.gatherer.get_traj_data(rob_ns)
                self.draw_trajectory(rob_ns, traj_data)
                self.draw_head_arrow(rob_ns, x, y, yaw)
                #self.draw_pose_covariance(rob_ns, x, y, cov_xy)
            # Retrieve grid keypoints (frontiers and landmarks)
            front_idx, xp_rat, known_cells = self.OGrid.locate_reachable_frontiers()
            lm_items = self.gatherer.get_landmarks_items()
            # Draw keypoints
            self.draw_landmarks(lm_items)
            self.draw_frontiers(front_idx)
            
            # Active control
            self.update_global_state_factors(xp_rat, lm_items, indiv_states)
            free_robs = [(rob, pose) for rob, (pose, _, status, _) in indiv_states.items() if status == "free_roaming"]
            #chosen_goals = []    # No task allocation
            #chosen_goals = self.frontier_only_goals(free_robs, indiv_states, lm_items, front_idx, xp_rat)
            chosen_goals = self.all_tasks_goals(free_robs, indiv_states, lm_items, front_idx, xp_rat, known_cells)
            if len(chosen_goals) > 0:
                for (rob_ns, _, (xG, yG)) in chosen_goals:
                    [x, y, yaw], w2od_tf, _, _ = indiv_states[rob_ns]
                    d_goal = ((xG - x)**2 + (yG - y)**2)**0.5
                    if d_goal > 1.0:    # Keep default "free_roaming" if goal is already to close
                        self.controller.send_control(rob_ns, w2od_tf, [(xG, yG)])
            else:
                for rob_ns, (_, w2od_tf, _, _) in indiv_states.items():
                    self.controller.send_control(rob_ns, w2od_tf, [])
        
        time = self.gatherer.get_time()
        # Record process indicators
        self.record_indicators(time, xp_rat, lm_items, indiv_states)
        # Update display
        self.root.title(f"Centralized Monitoring System (T = {time:.2f}  ;  Xp_r = {xp_rat:.2f})")
        self.root.after(self.delay, self.update)

    def record_indicators(self, time, xp_rat, lm_items, indiv_states):
        t_stamp = round(float(time), 2)
        if self._last["time"] == t_stamp:
            return
        X_score = xp_rat
        W_score = 0.
        V_score = 0.
        T_score = 0.
        N = len(indiv_states)
        for (_, _, _, w, _) in lm_items:
            W_score += w
            if w >= 2:
                V_score += 1
        for ns, ([x, y, _], _, _, tau) in indiv_states.items():
            T_score += tau/N
            if ns not in self._pathLen:
                self._pathLen[ns] = 0.
            S = self._pathLen[ns]
            x0, y0 = self._last[ns] if ns in self._last else x, y
            dS = ((x - x0)**2 + (y - y0)**2)**0.5
            self._pathLen[ns] = S + dS
            self._last[ns] = x, y
        self.indicators.append([t_stamp , X_score, W_score, V_score, T_score])
        self._last["time"] = t_stamp

    def export_results(self, res_folder = "sim_results"):
        try:
            os.mkdir(res_folder)
        except:
            pass
        csv_file = f'{res_forlder}/indicators'
        header = ['Time', 'X_score', 'W_score', 'V_score', 'T_score']
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in self.indicators:
                writer.writerow(row)
    
    def run(self):
        self.root.mainloop()


# ================= Main =================

def ros_spin(executor):
    executor.spin()

def main():
    rclpy.init()
    namespaces = get_namespaces()
    data_gath = DataGatherer(group_size = len(namespaces))
    controller = GlobalControlPublisher()
    executor = MultiThreadedExecutor()
    rob_nodes = []
    for n, rob_ns in enumerate(namespaces):
        isLeader = True if n == 0 else False    # Create a leader (frame director) for the group (typically 'rob1')
        rob_node = ObservationListener(data_gath, rob_ns, isLeader)
        rob_nodes.append(rob_node)
        executor.add_node(rob_node)
        controller.add_agent(rob_ns)
    executor.add_node(controller)
    try:
        # Start ROS2 processes in a background thread
        ros_thread = threading.Thread(target=ros_spin, args=(executor,), daemon=True)
        ros_thread.start()
        # Tkinter in main thread
        cms = CentralMonitor(data_gath, controller)
        cms.update()
        cms.run()   
    finally:
        for rob_node in rob_nodes:
            rob_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        cms.export_results()


if __name__ == "__main__":
    main()

