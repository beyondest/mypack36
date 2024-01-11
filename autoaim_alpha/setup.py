from setuptools import setup

package_name = 'autoaim_alpha'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ggbond',
    maintainer_email='liyuxuan12345678@outlook.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node_webcam         = autoaim_alpha:node_webcam:main',
            'my_node_img_processor  = autoaim_alpha.node_img_processor:main'
        ],
    },
)
