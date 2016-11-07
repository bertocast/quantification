#!/usr/bin/env python
# coding=utf-8
import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(name='quantification',
      version="0.0.1",
      author='Alberto Castaño',
      author_email="bertocast@gmail.com",
      description='Quantification algorithms',
      packages=find_packages(exclude='tests'),
      long_description=read('README.md'),
      keywords=['quantification', 'machine learning'],
      classifiers=[
          "Development Status :: 3 - Alpha",
          'Intended Audience :: Developers',
          'Natural Language :: English',
          'Operating System :: Unix',
          'Programming Language :: Python :: 2.7'
      ],
      include_package_data=True,
      install_requires=[
          'numpy',
          'sklearn',
          'pandas',
          'dispy'
      ],
      tests_require=[
          'nose',
      ]
      )