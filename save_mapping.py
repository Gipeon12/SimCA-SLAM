import os
import time
import rclpy
from rclpy.node import Node as rcl_Node
from launch import LaunchDescription, LaunchService
from launch.actions import GroupAction, ExecuteProcess


### SERVICE ASSETS ###

def get_mob_names():  # Nodes awareness
    rclpy.init()
    enum = rcl_Node("mob_enumerator")
    # Allow DDS discovery to complete
    time.sleep(0.5)
    # Get mob names
    nodes = enum.get_node_names_and_namespaces()
    mob_names = []
    for node, ns in nodes:
        name = ns[1:]
        if ns != "/" and not name in mob_names:
            mob_names.append(name)
    enum.destroy_node()
    rclpy.shutdown()
    return mob_names

def set_save_path(out_folder = "sim_results"):  # Create and return a directory to save maps
    try:
        os.mkdir(out_folder)
    except:
        pass
    finally:
        slamID = sum(1 for entry in os.scandir(out_folder)) + 1
        slam_folder = f"{out_folder}/slam{slamID}"
        os.mkdir(slam_folder)
        return os.path.join(os.getcwd(), slam_folder)


### SAVE ACTIONS ###

def group_save_actions(mob_names, save_path):
    save_actions = []
    for rob_ns in mob_names:
        
        save_proc = ExecuteProcess(cmd=['ros2', 'service', 'call',
                                        f'/{rob_ns}/slam_toolbox/save_map',
                                        'slam_toolbox/srv/SaveMap',
                                        f"{{name: {{data: '{save_path}/{rob_ns}_map'}}}}"],
                                   output='screen')
        
        save_actions += [save_proc,]
    return save_actions

def generate_save_service(mob_names, save_path):
    save_actions = group_save_actions(mob_names, save_path)
    # Create Launch Description
    ld = LaunchDescription([GroupAction(save_actions)])
    # Create Launch Service
    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls


### MAIN FUNCTION ###

def main():
    mob_names = get_mob_names()
    save_path = set_save_path()
    save = generate_save_service(mob_names, save_path)
    try:
        print(f"[SAVE-SERVICE] Run save service - target folder: {save_path}")
        save.run()
    finally:
        save.shutdown()
        print("[SAVE-SERVICE] Shutdown save service.")


if __name__ == "__main__":
    main()
