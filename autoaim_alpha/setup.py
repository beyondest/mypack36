from setuptools import setup,find_packages
import os
from glob import glob
package_name = 'autoaim_alpha'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='liyuxuan',
    maintainer_email='liyuxuan12345678@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_webcam_mv = autoaim_alpha.node_webcam_mv:main',
            'node_detector = autoaim_alpha.node_detector:main',
            'node_test = autoaim_alpha.node_test:main',
            'node_test2 = autoaim_alpha.node_test2:main',
            'node_test3 = autoaim_alpha.node_test3:main',
            'node_observer = autoaim_alpha.node_observer:main',
            'node_decision_maker = autoaim_alpha.node_decision_maker:main',
            'node_com = autoaim_alpha.node_com:main',
            'node_marker = autoaim_alpha.node_marker:main'
        ],
    },
)
