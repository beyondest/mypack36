from setuptools import setup,find_packages

package_name = 'autoaim_alpha'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'node_img_processor = autoaim_alpha.node_img_processor:main'
        ],
    },
)
