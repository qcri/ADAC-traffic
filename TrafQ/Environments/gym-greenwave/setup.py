#!/usr/bin/env python
from setuptools import setup
import setuptools

setup(name='gymGreenWave',
      version='0.0.1',
      #package_dir = {"gymGreenWave": "gymGreenWave"},
      #packages = ["gymGreenWave"],
      packages = setuptools.find_packages(),
      install_requires=['gym',
                        'pandas',
                        'cfg_load'],
    include_package_data=True
      )
