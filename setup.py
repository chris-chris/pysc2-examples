"""NSML Example for StarCraft 2
=====================================


Please edit this text to describe what this file does. It is included
automatically in the documentation.

"""
#nsml: chrisai/starcraft2-docker:latest

from distutils.core import setup
setup(
    name='nsml StarCraft2 MineralShards',
    version='0.1',
    description='NSML StarCraft2 CollectMineralShards',
    install_requires=[
        'visdom',
        'pillow',
        'numpy',
        'absl-py',
        'cloudpickle',
    ]
)
