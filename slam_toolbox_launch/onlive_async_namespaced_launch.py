import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.conditions import IfCondition
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration, AndSubstitution, NotSubstitution, TextSubstitution
from launch_ros.actions import LifecycleNode 
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from lifecycle_msgs.msg import Transition
from launch_ros.descriptions import ParameterFile
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Launch arguments
    namespace = LaunchConfiguration('namespace')
    slam_params_file = LaunchConfiguration('slam_params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    use_lifecycle_manager = LaunchConfiguration('use_lifecycle_manager')

    # Declare arguments
    declare_namespace_cmd = DeclareLaunchArgument(
        'namespace', default_value='rob1', description='Robot namespace'
    )
    declare_slam_params_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(
            get_package_share_directory("slam_toolbox"),
            'config', 'mapper_params_online_async.yaml'),
        description='SLAM Toolbox parameter file'
    )
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='true', description='Use simulation clock'
    )
    declare_autostart_cmd = DeclareLaunchArgument(
        'autostart', default_value='true', description='Automatically start node'
    )
    declare_use_lifecycle_cmd = DeclareLaunchArgument(
        'use_lifecycle_manager', default_value='false',
        description='Enable lifecycle manager'
    )

    # Remap topics
    remappings = [
        ('/map', 'map'),
        ('/tf', 'tf'), ('/tf_static', 'tf_static'),
        ('/map_metadata', 'map_metadata'),
        ('/scan', 'scan'),
    ]

    # Lifecycle SLAM Toolbox node
    start_async_slam_toolbox_node = LifecycleNode(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        namespace=namespace,
        parameters=[
            ParameterFile(slam_params_file, allow_substs=True),
            {'use_sim_time': use_sim_time,
             'use_lifecycle_manager': use_lifecycle_manager,}
        ],
        remappings=remappings,
        output='screen'
    )
    
    configure_event = EmitEvent(
        event=ChangeState(
          lifecycle_node_matcher=matches_action(start_async_slam_toolbox_node),
          transition_id=Transition.TRANSITION_CONFIGURE
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager)))
    )

    activate_event = RegisterEventHandler(
        OnStateTransition(
            target_lifecycle_node=start_async_slam_toolbox_node,
            start_state="configuring",
            goal_state="inactive",
            entities=[
                LogInfo(msg="[LifecycleLaunch] Slamtoolbox node is activating."),
                EmitEvent(event=ChangeState(
                    lifecycle_node_matcher=matches_action(start_async_slam_toolbox_node),
                    transition_id=Transition.TRANSITION_ACTIVATE
                ))
            ]
        ),
        condition=IfCondition(AndSubstitution(autostart, NotSubstitution(use_lifecycle_manager)))
    )

    # Launch description
    ld = LaunchDescription()
    ld.add_action(declare_namespace_cmd)
    ld.add_action(declare_slam_params_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_autostart_cmd)
    ld.add_action(declare_use_lifecycle_cmd)
    ld.add_action(start_async_slam_toolbox_node)
    ld.add_action(configure_event)
    ld.add_action(activate_event)

    return ld

