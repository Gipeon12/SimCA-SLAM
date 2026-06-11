# This file defines three classes to handle and run a live simulation of a 2D multi-robot system,
# involving the publication of odometry data, TF transforms and laser scans as ROS2 topics during runtime.
# Robots have the possibility to use a feedback function to reach target zones on the map.
# On-screen display can be enabled if necessary.

import os
import csv
import sys
import time
import rclpy
import threading
import math as m
import numpy as np
import tkinter as tk
from tkinter import Canvas
from tf2_ros import Buffer
from rclpy.node import Node
from PIL import Image, ImageTk
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan
from simca_interface.msg import ObservationSet, ControlSet
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from geometry_msgs.msg import Point, Vector3, Quaternion, TransformStamped



### SIMULATION ###

class LiveSim:

    def __init__(self, cohort, live_map, sim_params):
        # --- Simulation parameters ---
        self.Tstep = sim_params["Tstep"]
        self.EndTm = sim_params["EndTm"]
        self.Tsimu = 0.0
        # --- Robots and state manager ---
        self.Mobs = cohort
        self.MSManager = MultiStateManager(cohort, self.Tstep)
        # --- Mutual awareness ---
        for mob in self.Mobs:
            mob.setCohort(self.Mobs)
        # --- Display ---
        self.LiveMap = live_map
        self.onDisp = self.LiveMap.onDisp if self.LiveMap else False
        # --- Global exit flag ---
        self.EXIT = threading.Event()
    
    def runTimer(self):
        if self.Tsimu >= self.EndTm:
            self.EXIT.set()
            return
        self.Tsimu += self.Tstep
        if self.onDisp:
            self.LiveMap.updateDisplay(self.Tsimu)
    
    def _disp_timer(self):
        if self.EXIT.is_set():
            self.exitDisplay()
            return
        self.runTimer()
        self.LiveMap.root.after(int(self.Tstep * 1000), self._disp_timer)

    def _off_timer(self):
        try:
            while not self.EXIT.is_set():
                time.sleep(self.Tstep)
                self.runTimer()
        except KeyboardInterrupt:
            self.EXIT.set()

    def exitDisplay(self):
        if self.LiveMap and self.LiveMap.root:
            try:
                self.LiveMap.root.quit()
                self.LiveMap.root.destroy()
            except Exception:
                pass

    def spinExecution(self):
        exe = rclpy.executors.SingleThreadedExecutor()
        exe.add_node(self.MSManager)
        # =============================
        def spin_exe():
            try:
                print("[SIM-STATE] [NODES-EXEC] Spin executor.")
                while rclpy.ok() and not self.EXIT.is_set():
                    exe.spin_once(timeout_sec=self.Tstep)
            except KeyboardInterrupt:
                pass
            finally:
                print("[SIM-STATE] [NODES-EXEC] Shutdown executor.")
                self.MSManager.writePathProgress()
                exe.shutdown()
                self.MSManager.destroy_node()
        # =============================
        self.Thread = threading.Thread(target=spin_exe, daemon=False)
        self.Thread.start()

    def runSimulation(self):
        self.spinExecution()
        try:
            print("[SIM-STATE] [SIMULATION] Start simulation.")
            if self.onDisp:
                self.LiveMap.root.after(0, self._disp_timer)
                self.LiveMap.root.mainloop()
            else:
                self._off_timer()
        except KeyboardInterrupt:
            print("[SIM-STATE] [SIMULATION] Keyboard interruption.")
        finally:
            print("[SIM-STATE] [SIMULATION] End of simulation.")
            self.EXIT.set()
            if self.onDisp:
                self.exitDisplay()
            if hasattr(self, "Thread"):
                self.Thread.join()


### ENVIRONMENT ###

class LiveMap():
    
    def __init__(self, env_image = "546t1361", onDisplay = False):
        # Generate or retrieve the map image (should be in RGBA format)
        self.MapDir = f"arenas/{env_image}.png"
        self.MapImg = Image.open(self.MapDir)  # Load the image using Pillow
        self.Image = np.array(self.MapImg) # Convert to numpy array
        # Constants for the robot simulation
        self.Size = len(self.Image)
        self.MRGbrd = self.Size//20  # Distance maximal to be considered close to a border
        self.onDisp = onDisplay
        if self.onDisp:
            self.openDisplay()
        
    def openDisplay(self):
        self.root = tk.Tk()
        self.root.title("2D Multi Mapping GUI")
        self.root.geometry()
        # Convert to a PhotoImage
        self.BitMap = ImageTk.PhotoImage(self.MapImg)
        # Create left column for map and robot movement
        self.MapFrame = tk.Frame(self.root, bg = "#f0f0f0") # "#f0f0f0" is the default background color for the widget
        self.MapFrame.grid(row = 0, column = 0, padx = (10, 5), sticky = "nswe")
        self.MapCanvas = Canvas(self.MapFrame, width = self.Size, height = self.Size, bg = "white")
        self.MapCanvas.pack(pady = 10)
        # Store the image in a persistent variable
        self.MapCanvas.create_image(0, 0, anchor = "nw", image = self.BitMap)
        # Keep a reference to the image to avoid garbage collection
        self.MapCanvas.image = self.BitMap
        # Create a frame to display the local surrounding maps
        self.SurroundFrame = tk.Frame(self.root, bg = "#f0f0f0")
        self.SurroundFrame.grid(row = 0, column = 1, padx = (5, 10), sticky = "nswe")
        # Initialize a counter to track simulated time
        self.ChronoCanvas = Canvas(self.SurroundFrame, width = 200, height = 30, bg = "lightgrey")
        self.ChronoCanvas.pack(pady = 10)
        self.CHRONO = self.ChronoCanvas.create_text(52, 18, text = "TIME: 0.0 s", font = ('arial', 10, 'bold'))
        
    def updateDisplay(self, Tsimu):
        if self.onDisp:
            self.ChronoCanvas.itemconfig(self.CHRONO, text = f"TIME: {Tsimu:.2f} s")
        else:
            pass


### MULTI STATE MANAGER ###

class MultiStateManager(Node):

    def __init__(self, cohort, time_step):
        self._Cohort = cohort
        self._Tsimul = 0.0
        self._Tstep = time_step

        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        super().__init__('multi_state_manager')

        # Publishers
        self._ODpublishers = {}
        self._SCpublishers = {}
        self._TFpublishers = {}
        self._STpublishers = {}
        self._OBSpublishers = {}
        
        # Subscriptions (task acquisition)
        self._CTRLsubs = {}
        self._CTRLdata = {}
        
        # Messages
        self._ODmessages = {}     
        self._SCmessages = {}
        self._TFmessages = {}
        self._STmessages = {}
        
        for mob in self._Cohort:
            msg_params = mob.getMessageParams()
            rob_ns = msg_params['namespace']
            self._ODpublishers[rob_ns] = self.create_publisher(Odometry, f"{rob_ns}/odom", 10)
            self._SCpublishers[rob_ns] = self.create_publisher(LaserScan, f"{rob_ns}/scan", 10)
            self._TFpublishers[rob_ns] = self.create_publisher(TFMessage, f"{rob_ns}/tf", 10)
            self._STpublishers[rob_ns] = self.create_publisher(TFMessage, f"{rob_ns}/tf_static", qos)
            self._OBSpublishers[rob_ns] = self.create_publisher(ObservationSet, f"{rob_ns}/observation", 10)
            self._CTRLsubs[rob_ns] = self.create_subscription(ControlSet, f"{rob_ns}/control", lambda msg, ns=rob_ns: self.task_callback(ns, msg), 10)
            self._CTRLdata[rob_ns] = ([0., 0., 0.], [])
            od_msg, sc_msg, tf_msg, st_msg = self.initROSmessages(msg_params)
            self._ODmessages[rob_ns] = od_msg     
            self._SCmessages[rob_ns] = sc_msg
            self._TFmessages[rob_ns] = tf_msg
            self._STmessages[rob_ns] = st_msg
            # Publish static transforms
            self.publishRigidBase(rob_ns)

        # Timers
        self._timer = self.create_timer(time_step, self.stepProcess)
        
        # Path progress recording
        self._PROG = []

    def initROSmessages(self, msg_params):
        # Odometry
        od_msg = Odometry()
        od_msg.header.frame_id = "odom"
        od_msg.child_frame_id = "base_link"
        # Scan
        sc_msg = LaserScan()
        sc_msg.header.frame_id = "laser_frame"
        sc_msg.angle_min = msg_params["angle_min"]
        sc_msg.angle_max = msg_params["angle_max"]
        sc_msg.angle_increment = msg_params["angle_inc"]
        sc_msg.time_increment = 0.0
        sc_msg.scan_time = msg_params["time_step"]
        sc_msg.range_min = msg_params["range_min"]
        sc_msg.range_max = msg_params["range_max"]
        # TF message
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        # Static TF message
        st_msg = TFMessage()
        return od_msg, sc_msg, tf_msg, st_msg

    def task_callback(self, rob_ns, msg):
        w2od_tf = msg.w2od_tf
        wpts_x = msg.wpts_x
        wpts_y = msg.wpts_y
        waypts = [(x, y) for x, y in zip(wpts_x, wpts_y)]
        self._CTRLdata[rob_ns] = (w2od_tf, waypts)
    
    def publishRigidBase(self, rob_ns):
        ts_base = TransformStamped()
        ts_base.header.frame_id = "base_link"
        ts_base.child_frame_id = "laser_frame"
        ts_base.transform.translation = Vector3(x=0.0, y=0.0, z=0.0)
        ts_base.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self._STmessages[rob_ns].transforms.append(ts_base)
        self._STpublishers[rob_ns].publish(self._STmessages[rob_ns])
        self.get_logger().info(
            f"Published rigid base transform: {ts_base.header.frame_id} -> {ts_base.child_frame_id} "
            f"[trans=({ts_base.transform.translation.x:.3f}, {ts_base.transform.translation.y:.3f}, {ts_base.transform.translation.z:.3f}), "
            f"rot=({ts_base.transform.rotation.x:.3f}, {ts_base.transform.rotation.y:.3f}, {ts_base.transform.rotation.z:.3f}, {ts_base.transform.rotation.w:.3f})]"
        )
    
    def getPoseLocation(self, pose):
        x, y, z = pose.get("x", 0.0), pose.get("y", 0.0), pose.get("z", 0.0)
        return Point(x=x, y=y, z=z)
        
    def getPoseTranslation(self, pose):
        x, y, z = pose.get("x", 0.0), pose.get("y", 0.0), pose.get("z", 0.0)
        return Vector3(x=x, y=y, z=z)
    
    def getPoseQuaternion(self, pose):
        r, p, q = pose.get("roll", 0.0), pose.get("pitch", 0.0), pose.get("yaw", 0.0)
        cr, sr = m.cos(r/2), m.sin(r/2)
        cp, sp = m.cos(p/2), m.sin(p/2)
        cq, sq = m.cos(q/2), m.sin(q/2)
        x = sr*cp*cq - cr*sp*sq
        y = cr*sp*cq + sr*cp*sq
        z = cr*cp*sq - sr*sp*cq
        w = cr*cp*cq + sr*sp*sq
        return Quaternion(x=x, y=y, z=z, w=w)

    def setObservationMessage(self, t, od_pose, scan_angs, scan_rges, las_lim, status, mut_obs, vlm_obs):
        obs_msg = ObservationSet()
        # Odometric pose and scan
        obs_msg.sim_time    = t
        obs_msg.odom_pose   = [od_pose.get("x", 0.0), od_pose.get("y", 0.0), od_pose.get("yaw", 0.0)]
        obs_msg.scan_angles = scan_angs
        obs_msg.scan_ranges = scan_rges
        obs_msg.laser_limit = las_lim
        obs_msg.status = status
        # Mutual observations
        obs_msg.rob_names   = [obs[0] for obs in mut_obs]
        obs_msg.obs_ranges  = [obs[1][0] for obs in mut_obs]
        obs_msg.obs_angles  = [obs[1][1] for obs in mut_obs]
        # Virtual Landmark observation
        obs_msg.vlm_xs  = vlm_obs[0]
        obs_msg.vlm_ys  = vlm_obs[1]
        return obs_msg

    def publishLiveState(self, mobot, sim_time):
        rob_ns, status = mobot.getName(), mobot.getStatus()
        odom_pose, odom_covm = mobot.getEulerPose(), mobot.getCovar6Matrix()
        (scan_angles, scan_ranges), laser_limit = mobot.getScanValues(), mobot.getLaserLimit()
        mut_obs, vlm_obs = mobot.getRobObservations(sim_time)
        # update ODOMETRY message to comply with the conventional cartesian orientation (ROS REP 103 Standard Units of Measure and Coordinate Conventions)
        self._ODmessages[rob_ns].header.stamp = self.timestamp
        self._ODmessages[rob_ns].pose.pose.position = self.getPoseLocation(odom_pose)
        self._ODmessages[rob_ns].pose.pose.orientation = self.getPoseQuaternion(odom_pose)
        self._ODmessages[rob_ns].pose.covariance = odom_covm.flatten().tolist()
        # Update LASERSCAN message
        self._SCmessages[rob_ns].header.stamp = self.timestamp
        self._SCmessages[rob_ns].ranges = scan_ranges
        # Update TF message (odom -> base_link)
        self._TFmessages[rob_ns].header.stamp = self.timestamp
        self._TFmessages[rob_ns].transform.translation = self.getPoseTranslation(odom_pose)
        self._TFmessages[rob_ns].transform.rotation = self.getPoseQuaternion(odom_pose)
        tf_msg = TFMessage()
        tf_msg.transforms.append(self._TFmessages[rob_ns])
        obs_msg = self.setObservationMessage(sim_time, odom_pose, scan_angles, scan_ranges, laser_limit, status, mut_obs, vlm_obs)
        # Publish data
        if rclpy.ok():
            self._ODpublishers[rob_ns].publish(self._ODmessages[rob_ns])
            self._SCpublishers[rob_ns].publish(self._SCmessages[rob_ns])
            self._TFpublishers[rob_ns].publish(tf_msg)
            self._OBSpublishers[rob_ns].publish(obs_msg)
    
    def stepProcess(self):
        self.timestamp = self.get_clock().now().to_msg()
        prog = []
        for mob in self._Cohort:
            rob_ns = mob.getName()
            w2od_tf, waypts = self._CTRLdata[rob_ns]
            mob.runRobot(self._Tsimul, ctrl_data = (w2od_tf, waypts))
            self._CTRLdata[rob_ns] = (w2od_tf, [])    # IMPORTANT: delete the last target point once used (avoid repetitions)!
            prog += mob.getPathProg()
            self.publishLiveState(mob, self._Tsimul)
        self._PROG.append(prog)
        self._Tsimul += self._Tstep
    
    def writePathProgress(self, res_folder = "sim_results"):
        try:
            os.mkdir(res_folder)
        except:
            pass
        csv_file = f'{res_forlder}/path'
        with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for row in self._PROG:
                writer.writerow(row)


### VIRTUAL MANEUVER POINT (VMP) FOR ARTIFICIAL POTENTIAL FIELD (APF) DRIVING ###

class VirtualManeuverPoint:

    def __init__(self, rob_ns, rob_mass = 2.0, max_accel = 1.0, rob_radius = 0.16, vmp_xset = 0.2, lidar_range = 1.6, det_thresh = 0.8):
        self.name = rob_ns
        # Geometric parameters
        self.r_rob = rob_radius                                           # Robot radius (m)
        self.r_det = det_thresh                                           # Detection threshold for obstacle avoidance (m)
        self.r_lid = lidar_range                                          # Upper limit of the laser range (m)
        self.R_vmp = np.array([vmp_xset, 0., 0.])                         # VMP position vector in body frame
        # Yaw rotation matrix
        self.Mrot = lambda yaw: np.array([[np.cos(yaw), -np.sin(yaw), 0.], [np.sin(yaw), np.cos(yaw), 0.], [0., 0., 1.]])    
        # Dynamic parameters
        self.mass = rob_mass                                              # Robot mass (kg)
        self.zinr = self.mass * self.r_rob**2 / 2                         # Mass moment of inertia w/r to the vertical z-axis
        self.a_max = max_accel                                            # Max acceleration (m/s**2)
        # Calculation parameters
        self.k_asp = 0.10                                                 # Passage heading force coefficient
        self.k_rep = 0.05                                                 # Field repulsion coefficient
        self.k_att = 0.15                                                 # Target attraction coefficient
        self._sig = lambda x : 1 - np.exp(-10*x**2)                       # Repulsion cancellation when facing a free passage
        # Activities
        self.r_tar = 0.25                                                 # Arrival threshold around target (m)
        self.T_idle = 5.                                                  # Timeout for idle status (s)
        self.T_busy = 20.                                                 # Timeout for busy status (s)
        self.free_status()
    
    def free_status(self, ref_time = 0.):
        self.waypoints = []
        self.status = "free_roaming"
        self.ref_time = ref_time
    
    def idle_status(self, ref_time):
        self.waypoints = []
        self.status = "idle"
        self.ref_time = ref_time
        print(f"{self.name} has switched to idle phase (time: {ref_time:.2f}).")
    
    def set_new_goal(self, waypts, ref_time):
        # Ignore if a task is already going on
        if self.status == "free_roaming" and len(waypts) > 0:
            self.waypoints = waypts
            self.status = "busy"
            self.ref_time = ref_time
            print(f"New target ({waypts[-1][0]:.2f}, {waypts[-1][1]:.2f}) assigned to {self.name} (time: {ref_time:.2f}).")

    def get_vmp_position(self, rob_pose):
        [x, y, yaw] = rob_pose
        P_vmp = np.array([x, y, 0.]) + self.Mrot(yaw) @ self.R_vmp
        return P_vmp

    def smooth_curve(self, ranges):
        wrap_rges = np.concatenate([ranges[-2:], ranges, ranges[:2]])
        median_ranges = []
        for i in range(2, len(wrap_rges)-2):
            median_ranges += [float(np.median(wrap_rges[i-2:i+3]))]
        return np.array(median_ranges)

    def get_lidar_vectors(self, angles, ranges):
        vecs = []
        for ang, rge in zip(angles, ranges):
            if rge == float('inf'):
                continue
            vecs.append(rge * np.array([np.cos(ang), np.sin(ang), 0.]) - self.R_vmp)
        vecs = np.vstack(vecs) if len(vecs) > 0 else np.array([])
        return vecs

    def find_passage_in(self, lid_vectors):
        d_vecs = np.array([np.linalg.norm(v2-v1) for v1, v2 in zip(lid_vectors[:-1], lid_vectors[1:])])
        return np.where(d_vecs > 3*self.r_rob)[0]

    def get_passage_eval(self, v1, v2):
        v_mid, v2_c = (v1 + v2)/2, v2 * [1, -1, 1]                   # We define a conjugate vector for v2 (w/r x-axis)
        x_eval = v_mid[0]
        p_eval = np.linalg.norm(v2_c- v1)/np.linalg.norm(v2 - v1)    # We use the normalized distance between v1 and v2_c to evaluate the passage worthiness
        return x_eval, p_eval

    def get_circumcenter(self, v1, v2):
        vA, vB = v1[:2], v2[:2]
        d_AB = np.cross(v2, v1)[-1]
        vA_T, vB_T = vA[::-1]*[-1,1], vB[::-1]*[-1,1]    # Orthogonal 2D vectors
        a, b = np.linalg.norm(vA), np.linalg.norm(vB)
        vC = (a**2 * vB_T - b**2 * vA_T)/(2*d_AB)        # Position of the circumcenter given two vectors defining a triangle
        return np.append(vC, 0.)

    def get_waypoint_attraction(self, rob_pose, sim_time):
        if self.status != "busy":
            return np.zeros(3)
        f0 = self.mass * self.a_max
        # Get VMP position in world coordinates
        [x, y, yaw] = rob_pose
        P_vmp = self.get_vmp_position(rob_pose)
        # Get waypoint location in VMP frame
        x1, y1 = self.waypoints[0]
        P_att = np.array([x1, y1, 0.])
        R_att = self.Mrot(-yaw) @ (P_att - P_vmp)
        rho = np.linalg.norm(R_att) + 1e-3
        if rho < self.r_tar:
            self.waypoints = self.waypoints[1:]
            if len(self.waypoints) == 0:
                self.idle_status(ref_time = sim_time)    # Switch to idle phase once target is reached
        u_att = (1/rho) * R_att
        f_att = self.k_att * f0 * u_att
        return f_att

    def get_passage_aspiration(self, lid_vectors, isActive = True):
        f0, p_eval = self.mass * self.a_max, 1.0
        u_asp = np.array([1., 0., 0.])
        idx = self.find_passage_in(lid_vectors) if isActive else np.array([], dtype=np.int64)
        ok_pas = []
        for v1, v2 in zip(lid_vectors[idx], lid_vectors[idx+1]):
            x_eval, p_eval = self.get_passage_eval(v1, v2)
            if x_eval > 0.15:    # Only passages in the front half-plane are considered
                vC = self.get_circumcenter(v1, v2)
                ok_pas += [[p_eval, vC],]
        if len(ok_pas) > 0:
            p_eval, vC = sorted(ok_pas, key = lambda x : x[0])[0]
            u_asp = vC/np.linalg.norm(vC)
        f_asp = self.k_asp * f0 * u_asp
        return f_asp, p_eval

    def get_field_repulsion(self, lid_vectors):
        # Net force calculation in VMP frame
        nF, f0 = np.zeros(3), 0.
        for vec in lid_vectors:
            r = np.linalg.norm(vec)
            if r > self.r_det:
                continue
            f = (1/r - 1/self.r_det)**2
            if f > f0:
                f0 = f    # Keep intensity of the closest point
            nF -= f * vec/r
        # Repulsion force calculation
        nF_norm = np.linalg.norm(nF)
        u_rep = nF/nF_norm if nF_norm != 0 else np.zeros(3)
        f_rep = self.k_rep * f0 * u_rep
        return f_rep

    def get_driving_wrench(self, lid_vectors, rob_pose, sim_time):
        f_att = self.get_waypoint_attraction(rob_pose, sim_time)
        f_asp, p_eval = self.get_passage_aspiration(lid_vectors)
        f_rep = self.get_field_repulsion(lid_vectors)
        # Total driving force and resulting torque (force applied at VMP)
        f_drv = f_att + f_asp + self._sig(p_eval) * f_rep
        tau_drv = np.cross(self.R_vmp, f_drv)
        return f_drv, tau_drv

    def compute_wheels_command(self, angles, ranges, rob_pose, sim_time):
        smooth_rges = self.smooth_curve(ranges)
        lid_vectors = self.get_lidar_vectors(angles[::2], smooth_rges[::2])    # Sampling of scan values (step > 1 to reduce the computational cost)
        #   Timeout handling
        if self.status == "busy" and sim_time - self.ref_time > self.T_busy:
            print(f"[Timeout] {self.name} failed to reach target.")
            self.idle_status(ref_time = sim_time)
        elif self.status == "idle" and sim_time - self.ref_time > self.T_idle:
            self.free_status(ref_time = sim_time)
        #   Forces calculation
        f_drv, tau_drv = self.get_driving_wrench(lid_vectors, rob_pose, sim_time)
        fx, tau = float(f_drv[0]), float(tau_drv[2])
        #   Compute linear and angular accelerations
        aLIN = fx/self.mass
        aANG = tau/self.zinr
        #   Compute left and right wheels accelerations
        aLW = aLIN - aANG*self.r_rob
        aRW = aLIN + aANG*self.r_rob
        #   Normalization
        uLW = aLW/self.a_max
        uRW = aRW/self.a_max
        #   Auto-scale to input in [-1, 1]
        uMAX = max(abs(uLW), abs(uRW))
        (iLW, iRW) = (uLW/uMAX, uRW/uMAX) if uMAX != 0 else (0, 0)
        return iLW, iRW, lid_vectors


### OBSERVER TO HANDLE MUTUAL AND VLM (VIRTUAL LANDMARKS) OBSERVATIONS

class Observer:
    
    def __init__(self, section_cutoff = 0.3):
        self.d_cut = section_cutoff
        # Record Mutual Observations between agents
        self.MutObs = {}
        # Generate a Virtual Landmark (VLM) point from LiDAR scans
        self.vlm_ranges = []
        self.vlm_angles = []
        self.sectors = []
    
    def add_mutual_observation(self, mate_ns, ref_time, rho_obs, phi_obs):
        self.MutObs[mate_ns] = [ref_time, (rho_obs, phi_obs)]
    
    def add_scan_sectors(self, vectors):
        d_vecs = np.array([np.linalg.norm(v2-v1) for v1, v2 in zip(vectors[:-1], vectors[1:])])
        idx = np.where(d_vecs > self.d_cut)[0]
        i0 = idx[0] if len(idx) > 0 else -1
        reranked = np.append(vectors[i0 + 1:], vectors[:i0 + 1], axis=0)
        idx_sp = idx[1:] - i0
        self.sectors = np.split(reranked, idx_sp)
    
    def find_vlm(self):
        if len(self.sectors) < 5:
            return False
        v_bary = np.zeros(3)
        vlm_xs, vlm_ys = [], []
        N_vtx = max(1, len(self.sectors))
        # First iteration [from robot]: barycenter of closest points in each section
        for vecs in self.sectors:
            v_min = vecs[np.argmin(np.linalg.norm(vecs, axis=1))]
            v_bary += v_min/N_vtx
        # Second iteration [from previous barycenter]: same process to converge towards a more stable point
        for vecs in self.sectors:
            vtx = vecs[np.argmin(np.linalg.norm(vecs-v_bary, axis=1))]
            vlm_xs.append(vtx[0])
            vlm_ys.append(vtx[1])
        self.vlm_xs = vlm_xs
        self.vlm_ys = vlm_ys
        return True
    
    def get_observations(self, ref_time):
        mut_obs = [[rob_ns, data[1]] for rob_ns, data in self.MutObs.items() if abs(data[0]-ref_time) < 0.001]
        vlm_obs = [self.vlm_xs, self.vlm_ys] if self.find_vlm() else [[], []]
        return mut_obs, vlm_obs


### MOBILE ROBOT ###

class Mobot():
    
    def __init__(self, live_map, namespace, init_pose, sim_params, rob_params):
        # Live environment       
        self.LiveMap = live_map
        # Identification tag, number and color
        self.IDnum = rob_params["id_num"]
        self.RobName = f"rob{self.IDnum}" if namespace == None else namespace
        self.RobCol = rob_params["color_tag"]
        # Driving unit (Virtual Maneuver Point)
        self.vmp = VirtualManeuverPoint(rob_ns = self.RobName,
                                        rob_mass = rob_params["rob_mass"],
                                        max_accel = rob_params["max_accel"],
                                        rob_radius = rob_params["core_radius"],
                                        vmp_xset = rob_params["vmp_xset"],
                                        lidar_range = rob_params["laser_limit"],
                                        det_thresh = rob_params["detec_thresh"])
        # Observation unit to track mutual detections and virtual landmarks
        self.obs = Observer()
        self.Tstep = sim_params["Tstep"]  # Time step in seconds
        self.Dfreq = sim_params["Dfreq"]  # Data transmission frequency
        self.Scale = sim_params["Scale"]  # Scale factor to convert one pixel in meters
        # Scale converter
        self.toImg = lambda x, y : (int(self.LiveMap.Size/2 + x/self.Scale), int(self.LiveMap.Size/2 - y/self.Scale))
        # Geometrical parameters
        self.RobRad = rob_params["core_radius"]
        self.PxlRad = int(self.RobRad/self.Scale)
        self.GeoMat = np.array([[1/2, 1/2], [-0.5/self.RobRad, 0.5/self.RobRad]])
        # Robot footprint as boolean mask for collision detection
        self.FPsize = 2*self.PxlRad + 2
        ys, xs = np.ogrid[:self.FPsize,:self.FPsize]
        self.FPmask = (xs-(self.FPsize-1)/2)**2 + (ys-(self.FPsize-1)/2)**2 <= self.PxlRad**2
        # Dynamical parameters
        self.MaxSpd = rob_params["max_speed"]
        self.MaxAcc = rob_params["max_accel"]
        # Sensor parameters
        self.LasLim = rob_params["laser_limit"]
        self.PxlLim = int(self.LasLim/self.Scale)
        self.LidPts = rob_params["lidar_points"]
        self.ScRges = None
        ang_offset = 2*np.pi * (self.LidPts//2)/self.LidPts
        self.ScAngs = np.linspace(0, 2*np.pi, self.LidPts+1)[:-1] - ang_offset
        # Laser range in pixels
        self._PxlRge = list(range(self.PxlRad + 1, self.PxlLim + 4, 5))
        # Record initial pose
        self.pose0 = init_pose
        # Initialize real pose
        self.x_real = init_pose[0]
        self.y_real = init_pose[1]
        self.yaw_real = init_pose[2]
        # Initialize odometry
        self.x_odom = 0
        self.y_odom = 0
        self.yaw_odom = 0
        # Left and right wheel dynamics
        self.speedL = 0.
        self.speedR = 0.
        # Systematic errors for left and right wheels. Are unique to each robot and randomly selected for diversity
        self._SIGb = rob_params["bias_coeff"]*self.MaxSpd     # Systematic Bias
        self._SIGe = rob_params["error_coeff"]*self.MaxSpd    # Random Error
        self._SIGc = (self._SIGb**2 + self._SIGe**2)**0.5     # Combined Uncertainty
        self.errSpL = np.random.normal(scale = self._SIGb)/self.MaxSpd
        self.errSpR = np.random.normal(scale = self._SIGb)/self.MaxSpd
        # Covariance matrix
        self.CovMat = np.zeros([3, 3])
        # Data recording
        self._PATH = []
        self._ODOM = []
        self._SCAN = []
        self._COVA = []
        self._prog = [0.,0.]
        # Init display features if needed
        self.onDisp = self.LiveMap.onDisp
        if self.onDisp and self.LiveMap is not None:
            self.initDisplay()

    def initDisplay(self):
        # Initialize Lidar points representation
        self._LIDPTS = [None for ang in self.ScAngs]
        # Create robot representation, with a visual marker at its front
        i, j = self.toImg(self.x_real, self.y_real)
        self._robot = self.LiveMap.MapCanvas.create_oval(i-self.PxlRad, j-self.PxlRad, i+self.PxlRad, j+self.PxlRad, fill = self.RobCol)
        # Attache a bar to the body to show its front direction (i.e. body frame x-axis)
        self.PxlBar = 1.2 * self.PxlRad
        self._fmark = self.LiveMap.MapCanvas.create_line(i, j, i + self.PxlBar*m.cos(self.yaw_real), j - self.PxlBar*m.sin(self.yaw_real), width = 3)
        # Create middle column for environment data
        self._SRDcanvas = Canvas(self.LiveMap.SurroundFrame, width = 2*self.PxlLim, height = 2*self.PxlLim, bg = "gray")
        self._SRDcanvas.pack(pady = 10)       
        # Display robot data
        self._RDATAframe = tk.Frame(self.LiveMap.SurroundFrame, bg = "#f0f0f0")
        self._RDATAframe.pack()       
        self._YAWlabel = tk.Label(self._RDATAframe, text = "Yaw: 0.0°", font = ("Arial", 10), bg = "#f0f0f0")
        self._YAWlabel.pack()
        self._SPDlabel = tk.Label(self._RDATAframe, text = "Speed: 0.00 m/s", font = ("Arial", 10), bg = "#f0f0f0")
        self._SPDlabel.pack()

    def setCohort(self, mates):
        self.Cohort = mates  

    def pulseLidar(self):
        ScRges = []
        for k, ang in enumerate(self.ScAngs):
            rge = float('inf')    # Infinite value if no hit up to the laser range limit (no reflexion)
            if self.onDisp and self._LIDPTS[k] != None:
                self.LiveMap.MapCanvas.delete(self._LIDPTS[k])
            for dpx in self._PxlRge:
                x = self.x_real + dpx * self.Scale * m.cos(ang + self.yaw_real)
                y = self.y_real + dpx * self.Scale * m.sin(ang + self.yaw_real)
                i, j = self.toImg(x, y)
                if min(i, j) < 0 or max(i, j) > self.LiveMap.Size:
                    break
                elif self.LiveMap.Image[j][i][0] == 0 or self.hitMob(x, y):
                    while self.LiveMap.Image[j][i][0] == 0 or self.hitMob(x, y):
                        dpx -= 1
                        x = self.x_real + dpx * self.Scale * m.cos(ang + self.yaw_real)
                        y = self.y_real + dpx * self.Scale * m.sin(ang + self.yaw_real)
                        i, j = self.toImg(x, y)
                    rge = dpx * self.Scale
                    if self.onDisp:
                        self._LIDPTS[k] = self.LiveMap.MapCanvas.create_oval(i-2, j-2, i+2, j+2, fill="red")
                    break
            ScRges += [rge*np.random.normal(loc = 1, scale = 0.01),] # Adding noise to each measure (noise is proportional to distance)
        self._SCAN += [ScRges,]
        self.ScRges = np.array(ScRges)
    
    def getSurround(self):
        self._SRDcanvas.delete("all")
        i, j = self.toImg(self.x_real, self.y_real)
        croparray = self.LiveMap.Image[max(j - self.PxlLim - 1, 0):min(j + self.PxlLim + 1, self.LiveMap.Size), max(i - self.PxlLim - 1, 0):min(i + self.PxlLim + 1, self.LiveMap.Size)]
        dimJ, dimI = len(croparray), len(croparray[0])
        if i < self.PxlLim + 1: # If the robot gets too close to the border, we fill in the surrounding map with gray stripes
            addleft = np.array([[[128,128,128,255] for i in range(self.PxlLim + 1 - i)] for j in range(dimJ)])
            croparray = np.hstack((addleft,croparray))
            dimI = len(croparray[0])
        elif self.LiveMap.Size - i < self.PxlLim + 1:
            addright = np.array([[[128,128,128,255] for i in range(self.PxlLim + 1 - (self.LiveMap.Size - i))] for j in range(dimJ)])
            croparray = np.hstack((croparray,addright))
            dimI = len(croparray[0])
        if j < self.PxlLim + 1:
            addtop = np.array([[[128,128,128,255] for i in range(dimI)] for j in range(self.PxlLim + 1 - j)])
            croparray = np.vstack((addtop,croparray))
            dimJ = len(croparray)
        elif self.LiveMap.Size - j < self.PxlLim + 1:
            addbottom = np.array([[[128,128,128,255] for i in range(dimI)] for j in range(self.PxlLim + 1 - (self.LiveMap.Size - j))])
            croparray = np.vstack((croparray,addbottom))
            dimJ = len(croparray)
        croparray = croparray.astype(np.uint8) # Converting to the right format for ImageTk.PhotoImage
        robarray = np.full_like(croparray, 128)
        for j in range(2*self.PxlLim):
            for i in range(2*self.PxlLim):
                dpx = ((self.PxlLim-j)**2+(self.PxlLim-i)**2)**0.5
                if  dpx <= self.PxlLim:
                    alpha = m.atan2(self.PxlLim-j,self.PxlLim-i)
                    i_loc = int(self.PxlLim - dpx*m.sin(alpha + self.yaw_real))
                    j_loc = int(self.PxlLim + dpx*m.cos(alpha + self.yaw_real))
                    robarray[j_loc][i_loc] = croparray[j][i]
                else:
                    robarray[j][i][3] = 255
        imsurround = ImageTk.PhotoImage(Image.fromarray(robarray))
        self._SRDcanvas.image = imsurround
        self._SRDcanvas.create_image(0, 0, anchor="nw", image=imsurround)
        self._SRDcanvas.create_oval(self.PxlLim-self.PxlRad, self.PxlLim-self.PxlRad, self.PxlLim+self.PxlRad, self.PxlLim+self.PxlRad, fill = self.RobCol)
        self._SRDcanvas.create_line(self.PxlLim, self.PxlLim, self.PxlLim, self.PxlLim - self.PxlBar, width = 3)
        for n in range(len(self.Cohort)):
            if n+1 == self.IDnum:
                continue
            else:
                mob = self.Cohort[n]
                self.displayMob(mob)
    
    def hasCollision(self):
        i, j = self.toImg(self.x_real, self.y_real)
        local = self.LiveMap.Image[j - self.PxlRad - 1:j + self.PxlRad + 1, i - self.PxlRad - 1:i + self.PxlRad + 1]
        hit_mask = self.FPmask & np.any(local == 0, axis=-1)    # Convert local area to boolean mask (True if obstacle) and compare with the robot's footprint mask
        hasCol = True if hit_mask.any() else False
        return hasCol
    
    def runRobot(self, sim_time, ctrl_data):
        # Record odometry, real trajectory and covariance data
        self._PATH += [[self.x_real, self.y_real, self.yaw_real],]
        self._ODOM += [[self.x_odom, self.y_odom, self.yaw_odom],]
        self._COVA += [[self.CovMat[0][0],self.CovMat[1][1],self.CovMat[2][2],self.CovMat[0][1],self.CovMat[0][2],self.CovMat[1][2]],]
        if self.onDisp:  # Update environment display
            #self.getSurround()
            self._SRDcanvas.after(int(self.Tstep*1000), self.getSurround)
        # Pulse LIDAR
        self.pulseLidar()
        # Observe around
        self.watchMobs(sim_time)
        # Command wheel speed to avoid obstacles and the W/E and N/S borders
        w2od_tf, waypts = ctrl_data
        tf_pose = self.getCorrectedPose(w2od_tf)
        self.recordPathProg(tf_pose) # Record path progress
        self.vmp.set_new_goal(waypts, ref_time = sim_time)
        iLW, iRW, lid_vectors = self.vmp.compute_wheels_command(self.ScAngs, self.ScRges, tf_pose, sim_time)
        A, B = self.MaxAcc, self.MaxAcc/self.MaxSpd
        self.speedL += ( A*iLW - B*self.speedL ) * self.Tstep
        self.speedR += ( A*iRW - B*self.speedR ) * self.Tstep
        # Add errors (systematic bias + random error) to real wheel speed
        Lspd_real = self.speedL * (1 + self.errSpL + np.random.normal(scale = self._SIGe)/self.MaxSpd)
        Rspd_real = self.speedR * (1 + self.errSpR + np.random.normal(scale = self._SIGe)/self.MaxSpd)
        # Compute the estimated and real (with errors) twists
        (SPD_estm, OMG_estm) = self.getTwist(self.speedL, self.speedR)
        (SPD_real, OMG_real) = self.getTwist(Lspd_real, Rspd_real)
        # Forward the vectorized scans to the observation unit
        self.obs.add_scan_sectors(lid_vectors)
        # Check collisions with objects (if so: linear speed = 0)
        if self.hasCollision():
            self.speedL -= SPD_estm
            self.speedR -= SPD_estm
            SPD_estm, SPD_real = 0., 0.
        # Update estimated and real ORIENTATIONS
        self.yaw_odom += OMG_estm * self.Tstep
        self.yaw_real += OMG_real * self.Tstep
        # Calculate the increment in position from the current speed
        dPos_estm = SPD_estm * self.Tstep
        dPos_real = SPD_real * self.Tstep
        # Update odometry
        self.x_odom += dPos_estm * m.cos(self.yaw_odom)
        self.y_odom += dPos_estm * m.sin(self.yaw_odom)
        # Update real trajectory
        self.x_real += dPos_real * m.cos(self.yaw_real)
        self.y_real += dPos_real * m.sin(self.yaw_real)
        # Propagate the overall uncertainty
        self.propagUncertainty()
        if self.onDisp:  # Update robot representation
            i, j = self.toImg(self.x_real, self.y_real)
            self.LiveMap.MapCanvas.coords(self._robot, i-self.PxlRad, j-self.PxlRad, i+self.PxlRad, j+self.PxlRad)
            self.LiveMap.MapCanvas.coords(self._fmark, i, j, i+self.PxlBar*m.cos(self.yaw_real), j-self.PxlBar*m.sin(self.yaw_real))
            # Update labels
            self._YAWlabel.config(text = f"Yaw: {(self.yaw_real*180/np.pi)%360:.1f}°")
            self._SPDlabel.config(text = f"Speed: {SPD_real:.2f} m/s (L: {self.speedL:.2f} | R: {self.speedR:.2f})")
        
    def watchMobs(self, sim_time):    # Record mutual observations between agents
        for n in range(len(self.Cohort)):
            if n+1 == self.IDnum:
                continue
            else:
                mob = self.Cohort[n]
                mob_ns = mob.getName()
                pmob = mob.getRealPose()
                vmob = pmob[:2] - np.array([self.x_real, self.y_real])
                if np.linalg.norm(vmob) < self.LasLim:
                    rho_obs = np.linalg.norm(vmob)
                    phi_obs = np.atan2(vmob[1], vmob[0]) -  self.yaw_real
                    self.obs.add_mutual_observation(mob_ns, sim_time, rho_obs, phi_obs)
    
    def hitMob(self, x, y):    # Check whether the laser beam has hit a robot
        for n in range(len(self.Cohort)):
            if n+1 == self.IDnum:
                continue
            else:
                mob = self.Cohort[n]
                pmob = mob.getRealPose()
                rmob = mob.getRobRadius()
                toMob = pmob[:2] - np.array([x, y])
                if np.linalg.norm(toMob) <= rmob:
                    return True
        return False

    def displayMob(self, mob):    # Display the other robots on the local window if they are close enough
        cmob = mob.getRobColor()
        pmob = mob.getRealPose()
        rmob = mob.getRobRadius()
        pxlr = int(rmob/self.Scale)
        relP = pmob[:2] - np.array([self.x_real, self.y_real])
        norm = np.linalg.norm(relP)
        if norm < self.LasLim - rmob:
            # Compute relative coordinates of the targeted robot in the local frame
            alpha = m.atan2(relP[0], relP[1])
            i_rel = int((self.LasLim - norm*m.cos(alpha + self.yaw_real))/self.Scale)
            j_rel = int((self.LasLim - norm*m.sin(alpha + self.yaw_real))/self.Scale)
            self._SRDcanvas.create_oval(i_rel-pxlr, j_rel-pxlr, i_rel+pxlr, j_rel+pxlr, fill = cmob)

    def getTwist(self, Lspd, Rspd):
        [V, Omega] = np.dot(self.GeoMat, np.array([Lspd, Rspd]))
        return V, Omega

    def getTwistUncertainty(self):
        sigL = abs(self.speedL)/self.MaxSpd*self._SIGc
        sigR = abs(self.speedR)/self.MaxSpd*self._SIGc
        Qw = np.array([[sigL**2, 0], [0, sigR**2]])
        M = np.dot(Qw, np.transpose(self.GeoMat))
        Qv = np.dot(self.GeoMat, M)
        return Qv

    def getJacobian(self):
        (V, _) = self.getTwist(self.speedL, self.speedR)
        Jp = np.array([[1, 0, -V*self.Tstep*m.sin(self.yaw_odom)], [0, 1, V*self.Tstep*m.cos(self.yaw_odom)], [0, 0, 1]])
        Ju = self.Tstep * np.array([[m.cos(self.yaw_odom), 0], [m.sin(self.yaw_odom), 0], [0, 1]])
        return Jp, Ju

    def propagUncertainty(self):
        (Jp, Ju) = self.getJacobian()
        Qv = self.getTwistUncertainty()
        Sp = np.dot(Jp, np.dot(self.CovMat, np.transpose(Jp)))
        Su = np.dot(Ju, np.dot(Qv, np.transpose(Ju)))
        self.CovMat = Sp + Su
    
    def getEulerPose(self):
        return {"x":self.x_odom,"y":self.y_odom,"z":0.0,"roll":0.0,"pitch":0.0,"yaw":self.yaw_odom}

    def getCovar6Matrix(self):
        cMat = np.zeros((6, 6))
        cMat[0, 0], cMat[0, 1], cMat[0, 5] = self.CovMat[0][0], self.CovMat[0][1], self.CovMat[0][2]
        cMat[1, 0], cMat[1, 1], cMat[1, 5] = self.CovMat[0][1], self.CovMat[1][1], self.CovMat[1][2]
        cMat[5, 0], cMat[5, 1], cMat[5, 5] = self.CovMat[0][2], self.CovMat[1][2], self.CovMat[2][2]
        return cMat

    def getScanValues(self):
        return self.ScAngs, self.ScRges

    def getInitialPose(self):
        return self.pose0
    
    def getOdomPose(self):
        return np.array([self.x_odom, self.y_odom, self.yaw_odom])
    
    def getCorrectedPose(self, w2od_tf):
        [x0, y0, yaw0] = w2od_tf
        # Transform to world coordinates
        x = x0 + self.x_odom*np.cos(yaw0) - self.y_odom*np.sin(yaw0) 
        y = y0 + self.x_odom*np.sin(yaw0) + self.y_odom*np.cos(yaw0)
        yaw = yaw0 + self.yaw_odom
        return np.array([x, y, yaw])
    
    def getRealPose(self):
        return np.array([self.x_real, self.y_real, self.yaw_real])
    
    def getRobObservations(self, sim_time):
        return self.obs.get_observations(sim_time)
    
    def getIDnum(self):
        return self.IDnum
        
    def getName(self):
        return self.RobName
    
    def getStatus(self):
        return self.vmp.status
    
    def getRobColor(self):
        return self.RobCol
    
    def getRobRadius(self):
        return self.RobRad
    
    def getLaserLimit(self):
        return self.LasLim
    
    def getLidarPointsNumber(self):
        return self.LidPts
    
    def getSpeedErrors(self):
        return [self.errSpL, self.errSpR]
    
    def getInitialPose(self):
        return self.pose0
    
    def getPathRecord(self):
        return self._PATH
    
    def getOdometryRecord(self):
        return self._ODOM
    
    def getScansRecord(self):
        return self._SCAN
    
    def getCovarianceRecord(self):
        return self._COVA

    def getMessageParams(self):
        msg_params = {"namespace": self.RobName,
                      "time_step": self.Tstep,
                      "data_freq": self.Dfreq,
                      "angle_min": float(self.ScAngs[0]),
                      "angle_max": float(self.ScAngs[-1]),
                      "angle_inc": 2*np.pi/self.LidPts,
                      "range_min": self.RobRad,
                      "range_max": self.LasLim,}
        return msg_params
    
    def recordPathProg(self, tf_pose):
        if len(self._PATH) < 2:
            return
        [x0, y0, _] = self._PATH[-2]
        [x1, y1, _] = self._PATH[-1]
        [x_tf, y_tf, _] = tf_pose
        S = self._prog[0]
        dS = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        Err = ((x_tf - x1)**2 + (y_tf - y1)**2)**0.5
        self._prog = [S + dS, Err]
        
    def getPathProg(self):
        return self._prog
