"""Start the R20 TWD serial bridge and optional hold-to-drive keyboard window."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB0',
                              description='Jetson USB bridge or UART device; prefer /dev/serial/by-id/...'),
        DeclareLaunchArgument('baud', default_value='230400'),
        DeclareLaunchArgument('robot_type', default_value='r20_twd'),
        DeclareLaunchArgument('keyboard', default_value='true',
                              description='Open local keyboard GUI; false for a separate terminal controller'),
        DeclareLaunchArgument('enable_motion', default_value='true',
                              description='false sends only stop packets for communication checks'),
        DeclareLaunchArgument('speed', default_value='0.10'),
        DeclareLaunchArgument('turn', default_value='0.30'),
        DeclareLaunchArgument('command_timeout', default_value='0.30'),
        DeclareLaunchArgument('feedback_timeout', default_value='0.50'),
        DeclareLaunchArgument('startup_timeout', default_value='10.0'),
    ]

    def param(name, kind):
        return ParameterValue(LaunchConfiguration(name), value_type=kind)

    bridge = Node(
        package='snail_hardware', executable='r20_base_driver', namespace='r20',
        output='screen', parameters=[{
            'port': param('port', str), 'baud': param('baud', int),
            'robot_type': param('robot_type', str),
            'enable_motion': param('enable_motion', bool),
            'command_timeout': param('command_timeout', float),
            'feedback_timeout': param('feedback_timeout', float),
            'startup_timeout': param('startup_timeout', float),
            'use_sim_time': False,
        }],
    )
    keyboard = Node(
        package='snail_hardware', executable='r20_keyboard', namespace='r20',
        output='screen', condition=IfCondition(LaunchConfiguration('keyboard')),
        parameters=[{'mode': 'gui', 'speed': param('speed', float),
                     'turn': param('turn', float), 'use_sim_time': False}],
    )
    # Ending either process ends the session and lets the bridge send final stops.
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason=reason))],
    )) for process, reason in (
        (bridge, 'R20 bridge exited'), (keyboard, 'R20 keyboard closed'),
    )]
    return LaunchDescription(arguments + handlers + [bridge, keyboard])
