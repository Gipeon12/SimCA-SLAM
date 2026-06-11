import os
import json
import rclpy
import argparse
import math as m
import numpy as np
import pandas as pd
from livesim_assets import LiveSim, LiveMap, Mobot


### ARGUMENTS MANAGEMENT ###

# Get simulation parameters from arguments
parser = argparse.ArgumentParser(allow_abbrev = False)
parser.add_argument("--offdisp", action='store_false', help = "Deactivate the live map dislay")
parser.add_argument("--nrobots", type = int, default = 1, help = "Number of robots running in the simulation, if no poses are provided by hand")
parser.add_argument("--initpose", nargs = 3, type = float, default=[], action='append', metavar=('x', 'y', 'yaw'), help = "Initial pose of a robot on the map (m, rad)")
parser.add_argument("--stoptime", type = float, default = 180.0, help = "End time for the simulation (in seconds)")
parser.add_argument("--envimage", type = str, default = "546t1361", help = "Name (seed) of the 2D map that will be used for the simulation")
args = parser.parse_args()


### GENERAL PARAMETERS ###

# Simulation Parameters
SIMU = {"Tstep":0.1, "Dfreq":0.5, "EndTm":args.stoptime, "Scale":0.02}


### SIMULATION FUNCTIONS ###

def define_initial_poses(N = args.nrobots):
    init_poses = []
    if N == 1:
        return init_poses + [[0., 0., 0.],]
    for k in range(N):
        yaw = k * 2*np.pi/N
        x = 0.5 * m.cos(yaw)
        y = 0.5 * m.sin(yaw)
        init_poses += [[x, y, yaw],]
    return init_poses

def load_config(id_num = 1):
    df = pd.read_csv("cohort_config.csv", skipinitialspace=True)
    rob_df = df[df["id_num"] == id_num]
    rob_cfg = rob_df.iloc[0].apply(lambda x: x.strip() if isinstance(x, str) else x).to_dict()
    return rob_cfg

def get_details(init_poses):
    rob_details = []
    for i, pose in enumerate(init_poses):
        rob_ns = f"rob{i+1}"
        rob_cfg = load_config(id_num = i+1)
        detail = {"name": rob_ns, "pose": {"x": pose[0], "y": pose[1], "yaw": pose[2]}, "params": rob_cfg}
        rob_details += [detail,]
    return rob_details

def set_live_environment(args = args):
    # Create a Live Map
    live_map = LiveMap(env_image = args.envimage, onDisplay = args.offdisp)
    ### STANDARD INITIALIZATION: If no poses are provided, a default distribution is used for the given number of robots
    size = live_map.Size
    init_poses = define_initial_poses() if len(args.initpose) == 0 else args.initpose
    rob_details = get_details(init_poses)
    return live_map, init_poses, rob_details

def init_live_simulation(live_map, init_poses, rob_details, sim_params = SIMU, on_disp = args.offdisp):
    # Create a Cohort
    sim_cohort = []
    for i, pose in enumerate(init_poses):
        rob_ns, init_pose, rob_params  = rob_details[i]["name"], rob_details[i]["pose"], rob_details[i]["params"]
        mob = Mobot(live_map = live_map, namespace = rob_ns, init_pose = pose, sim_params = sim_params, rob_params = rob_params)
        sim_cohort += [mob,]
    # Create a Simulation
    live_sim = LiveSim(cohort = sim_cohort, live_map = live_map, sim_params = sim_params)
    return live_sim


### MAIN FUNCTION ###

def main():
    try:
        live_map, init_poses, rob_details = set_live_environment()
        rclpy.init()
        print("[GUI-STARTER] Initialize simulation.")
        sim = init_live_simulation(live_map, init_poses, rob_details)
        input("[GUI-STARTER] [WAITING...] Press ENTER to start >> ")
        sim.runSimulation()
    finally:
        print("[GUI-STARTER] Exit simulation.")
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

