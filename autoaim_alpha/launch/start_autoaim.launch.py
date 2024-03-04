from launch import LaunchDescription
from launch_ros.actions import Node
package_name = 'autoaim_alpha'
def generate_launch_description():
    return LaunchDescription([
        Node(
            package=package_name,
            node_executable='node_webcam_mv',
            node_name='node_webcam_mv'
        ),
        Node(
            package=package_name,
            node_executable='node_com',
            node_name='node_com'
        ),
        Node(
            package=package_name,
            node_executable='node_detector',
            node_name='node_detector'
        ),
        Node(
            package=package_name,
            node_executable='node_observer',
            node_name='node_observer'
        ),
        Node(
            package=package_name,
            node_executable='node_decision_maker',
            node_name='node_decision_maker'
        ),
        Node(
            package=package_name,
            node_executable='node_marker',
            node_name='node_marker'
        ),
        Node(
            package='rviz2',
            node_executable='rviz2',
            node_name='rviz2'
        )

    ])
