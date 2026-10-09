"""Launch a bottle approach with an exclusive base-command topic."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    specifications = {
        'model_path': ('', 'Absolute path to the prepared yolov8n ONNX model'),
        'enable_motion': ('false', 'false verifies detection without wheel motion'),
        'start_base': ('true', 'Own the STM32 serial port; stop the keyboard launch first'),
        'start_camera': ('false', 'Start RealSense only if no camera driver is already running'),
        'port': ('/dev/ttyUSB0', 'Actual STM32 serial device on Jetson'),
        'stop_distance': ('0.40', 'Bottle clearance from robot front, in metres'),
        'camera_to_front': ('0.0', 'Camera-to-front distance, positive if camera is behind bumper'),
        'max_linear': ('0.08', 'Maximum forward speed in m/s'),
        'max_angular': ('0.30', 'Maximum turn speed in rad/s'),
        'frame_timeout': ('0.50', 'Maximum frame/result age in seconds'),
        'confidence': ('0.55', 'Bottle model confidence threshold'),
        'color_topic': ('/camera/camera/color/image_raw', 'RealSense RGB image'),
        'depth_topic': ('/camera/camera/aligned_depth_to_color/image_raw', 'Depth aligned to color'),
        'camera_info_topic': ('/camera/camera/aligned_depth_to_color/camera_info', 'Aligned intrinsics'),
        'cmd_topic': ('/bottle_approach/cmd_vel', 'Exclusive approach command topic'),
        'connected_topic': ('/r20/connected', 'Fresh telemetry status from the base bridge'),
        'require_base_connected': ('true', 'Require base telemetry status before motion'),
    }
    arguments = [DeclareLaunchArgument(key, default_value=value, description=description)
                 for key, (value, description) in specifications.items()]

    def param(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(PathJoinSubstitution([
            FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py'])),
        condition=IfCondition(LaunchConfiguration('start_camera')),
        launch_arguments={'enable_color': 'true', 'enable_depth': 'true',
                          'align_depth.enable': 'true', 'enable_sync': 'true'}.items(),
    )
    bridge = Node(
        package='snail_hardware', executable='r20_base_driver', namespace='r20',
        condition=IfCondition(LaunchConfiguration('start_base')), output='screen',
        remappings=[('cmd_vel', LaunchConfiguration('cmd_topic'))],
        parameters=[{'port': param('port', str), 'robot_type': 'r20_twd',
                     'enable_motion': param('enable_motion', bool), 'use_sim_time': False}],
    )
    kinds = {'model_path': str, 'enable_motion': bool, 'stop_distance': float,
             'camera_to_front': float, 'max_linear': float, 'max_angular': float,
             'frame_timeout': float, 'confidence': float, 'color_topic': str,
             'depth_topic': str, 'camera_info_topic': str, 'cmd_topic': str,
             'connected_topic': str, 'require_base_connected': bool}
    approach = Node(
        package='snail_bottle_approach', executable='bottle_approach', output='screen',
        parameters=[dict({key: param(key, kind) for key, kind in kinds.items()}, use_sim_time=False)],
    )
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=node, on_exit=[EmitEvent(event=Shutdown(reason=reason))]))
        for node, reason in ((bridge, 'R20 base exited'), (approach, 'Bottle approach exited'))]
    return LaunchDescription(arguments + handlers + [camera, bridge, approach])
