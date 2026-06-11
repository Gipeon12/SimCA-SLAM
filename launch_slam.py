import os
import sys
import yaml
import time
import rclpy
from rclpy.node import Node as rcl_Node
from launch import LaunchDescription, LaunchService
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


### SERVICE ASSETS ###

def get_mob_names():
    rclpy.init()
    enum = rcl_Node("mob_enumerator")
    # Allow DDS discovery to complete
    time.sleep(0.5)
    topics = enum.get_topic_names_and_types()
    mob_names = []
    for topic_name, _ in topics:
        parts = topic_name.split('/')[1:]
        if len(parts) > 1 and parts[1] == 'tf':  # look for all published '/tf'
            mob_names.append(parts[0])
    enum.destroy_node()
    rclpy.shutdown()
    return mob_names


### LIVE ACTIONS ###

def group_slam_actions(mob_names, yaml_path):
    # Set path to the SLAM Toolbox launch file
    slam_toolbox_dir = get_package_share_directory('slam_toolbox')
    #slam_launch_file = os.path.join(slam_toolbox_dir, 'launch', 'onlive_async_decentralized_multirobot_launch.py')
    slam_launch_file = os.path.join(slam_toolbox_dir, 'launch', 'onlive_async_namespaced_launch.py')
    # Declare the YAML params file
    slam_params_path = DeclareLaunchArgument('slam_params_file', default_value=yaml_path, description='Full path to the YAML params file')
    slam_params_file = LaunchConfiguration('slam_params_file')
    slam_actions = [slam_params_path,]
    for rob_ns in mob_names:
        
        # SLAM Toolbox launch description
        slam_toolbox_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource(slam_launch_file),
                                                       launch_arguments={'namespace': rob_ns, 'slam_params_file': slam_params_file}.items())
        
        slam_actions += [slam_toolbox_launch,]
    return slam_actions

def group_rviz_actions(mob_names, rviz_path):
    rviz_actions = []
    for rob_ns in mob_names:
        
        # Set RViz configuration
        rviz_node = Node(package='rviz2',
                         executable='rviz2',
                         namespace=rob_ns,
                         name='rviz',
                         output="screen",
                         remappings=[('/map', f'/{rob_ns}/map'),
                                     ('/map_updates', f'/{rob_ns}/map_updates'),
                                     ('/scan', f'/{rob_ns}/scan'),
                                     ('/tf', f'/{rob_ns}/tf'), ('/tf_static', f'/{rob_ns}/tf_static'),
                                     ('/clicked_point', f'/{rob_ns}/clicked_point'),
                                     ('/initialpose', f'/{rob_ns}/initialpose'),
                                     ('/goal_pose', f'/{rob_ns}/goal_pose'),],
                         arguments=['-d', rviz_path])
        
        rviz_actions += [rviz_node,]
    return rviz_actions

def generate_slam_service(mob_names):
    # Path to RViz and YAML config file
    rviz_path = os.path.join(os.path.dirname(__file__), 'rviz_config.rviz')
    yaml_path = os.path.join(os.path.dirname(__file__), 'slam_params.yaml')
    # Group Actions
    slam_actions = group_slam_actions(mob_names, yaml_path)
    rviz_actions = group_rviz_actions(mob_names, rviz_path)
    # Create Launch Description
    ld = LaunchDescription([GroupAction(slam_actions + rviz_actions)])
    # Create Launch Service
    ls = LaunchService()
    ls.include_launch_description(ld)
    return ls


### MAIN FUNCTION ###

def main():
    mob_names = get_mob_names()
    slam = generate_slam_service(mob_names)
    try:
        print("[SLAM-SERVICES] Run live services.")
        slam.run()
    except KeyboardInterrupt:
        print("[SLAM-SERVICES] Got interruption signal.")
    finally:
        print("[SLAM-SERVICES] Shutdown live services.")
        slam.shutdown()


if __name__ == "__main__":
    main()

