## Create a python environment with the necessary packages
        
    python3 -m venv ~/.venvs/simca-env
    source "$HOME/.venvs/simca-env/bin/activate"
    pip install matplotlib numpy scipy pandas plotly perlin-noise trimesh setuptools pyyaml gtsam tqdm networkx lark
    deactivate
    echo 'source ~/.venvs/simca-env/bin/activate' > .envrc
    direnv allow

## Create a custom package 'simca_interface' to hold the 'ObservationSet' and 'ControlSet' messages
        
In the usual ROS2 workspace:

    ros2 pkg create simca_interface --build-type ament_cmake

Paste ObservationSet.msg and ControlSet.msg in a new /msg folder, and update CMakeLists.txt accordingly.
Then build:

    colcon build --symlink-install --packages-select simca_interface --allow-overriding simca_interface

## Simulation-based Collaborative Active SLAM workspace

Simulated Collaborative Active SLAM => SimCA-SLAM (simca_slam)


### Workspace composition:

Parent folder: /SimCA-SLAM
    
/arenas                     Folder where are saved the newly generated arenas (2D environment for the simulation).
generate_arena2d.py         Generates a new random arena for the simulation. Uses Perlin noise for the obstacle distribution.

cohort_config.csv           Configuration file that defines every parameters for each robot in the cohort.
livesim_assets.py           Defines all objects featured during simulation time.
start_livesim.py            Loads parameters, initializes and launches the simulated environment.

rviz_config.rviz            RViz startup configuration file.
slam_params.yaml            Parameters that will be used by SLAM Toolbox for each SLAM process.
launch_slam.py              Launches all individual SLAM instances in parallel. 

launch_cmon.py              Launches the centralized monitoring system on top of the SLAM processes.

/sim_results                Folder where are saved the mapping results.
save_mapping.py             Saves all generated maps at the end of the SLAM processes.

/simca_interface            Must be built as an external package (in the ROS2 workspace) to hold the 'ObservationSet' and 'ControlSet' messages.
/slam\_toolbox\_launch      The file it contains must be copied to the ‘launch’ folder of the slam_toolbox package (built from source).


### Protocol to launch a simulation:

In four different terminal:

(T1) Execute start_livesim.py with the desired parameters.
This action will create the Mob State Manager nodes and publish the related topics within relevant namespaces (tf_static, tf, odom, scan).
If no arguments are passed:
- The 2D scene is displayed (--offdisp to deactivate).
- Only one robot spawns in the environment (--nrobots {N} to add up to 6 robots; add lines to cohort_config.csv for more).
- A default arena is used (--envimage {map_seed} to load another existing environment from the /arenas folder).
- The simulation will stop after 180 seconds of simulated time (--stoptime {T} to change).
- The robots evenly spawn around the center of the arena (--initpose {x y yaw} to define new initial poses; each call for this argument will add a new robot).
The simulation is initialized and paused until the 'Enter' key is pressed (last action to perform at startup).

(T2) Execute launch_slam.py to launch an individual SLAM process for each robot in the simulation.
This action detects the different namespaces, subscribes to the corresponding topics and starts a SLAM Toolbox instance within each namespace.
The robots will perform SLAM autonomously, using both their own collected data (LiDAR scans) and a common pool of localized scans.
The shared topic will be receiving from the robots some of their scans associated with a global pose estimation.

(T3) Execute launch_cmon.py to launch a centralized monitor that will use the shared topic to build a low resolution occupancy grid of the global environment.
This occupancy map will be used as a Centralized Monitoring System for decision making and task allocation using Expectation-Maximization.

(T1) Press 'Enter' in the first terminal to launch the simulation and start collecting data.
