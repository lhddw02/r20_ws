from glob import glob

from setuptools import find_packages, setup

package_name = 'snail_hardware'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Snail Robot Project',
    maintainer_email='maintainer@example.com',
    description='ROS 2 serial and keyboard tools for the R20 TWD prototype.',
    license='LicenseRef-Project-Proprietary',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'r20_base_driver = snail_hardware.base_driver:main',
        'r20_keyboard = snail_hardware.keyboard:main',
        'r20_serial_probe = snail_hardware.serial_probe:main',
    ]},
)
