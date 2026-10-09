from glob import glob
from setuptools import find_packages, setup

package_name = 'snail_bottle_approach'
setup(
    name=package_name, version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md', 'requirements-jetson.txt']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/scripts', glob('scripts/*.py') + glob('scripts/*.sh')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='Snail Robot Project', maintainer_email='maintainer@example.com',
    description='RGB-D bottle detection and approach for the R20 prototype.',
    license='LicenseRef-Project-Proprietary', tests_require=['pytest'],
    entry_points={'console_scripts': ['bottle_approach = snail_bottle_approach.node:main']},
)
