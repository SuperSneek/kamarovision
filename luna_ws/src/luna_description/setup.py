from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'luna_description'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'resource/xacro/meshes'), glob(os.path.join('resource', 'xacro/meshes/*.stl'))),
        (os.path.join('share', package_name, 'resource/xacro/collisions'), glob(os.path.join('resource', 'xacro/collisions/*.stl'))),
        (os.path.join('share', package_name, 'resource/xacro'), glob(os.path.join('resource', 'xacro/*.xacro'))),
        (os.path.join('share', package_name, 'resource/rviz'), glob(os.path.join('resource', 'rviz/*.rviz'))),
        (os.path.join('share', package_name, 'resource/urdf'), glob(os.path.join('resource', 'urdf/*.urdf'))),
        (os.path.join('share', package_name, 'resource/urdf'), glob(os.path.join('resource', 'urdf/*.xacro'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'meshes'), glob(os.path.join('meshes', '*.stl'))),
        (os.path.join('share', package_name, 'meshes/3d_modelle/Left_leg'), glob(os.path.join('meshes/3d_modelle/Left_leg', '*.stl'))),
        (os.path.join('share', package_name, 'meshes/3d_modelle/Right_leg'), glob(os.path.join('meshes/3d_modelle/Right_leg', '*.stl'))),
        (os.path.join('share', package_name, 'meshes/3d_modelle'), glob(os.path.join('meshes/3d_modelle', '*.stl'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jonas',
    maintainer_email='joecarverde@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
