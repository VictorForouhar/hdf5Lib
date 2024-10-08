#!/usr/bin/env python

from distutils.core import setup

setup(name='hdf5Lib',
      version='2022.8',
      description='Python3 package used for paralell reading of HDF5 files that have been split into several files.',
      author='Victor Forouhar Moreno',
      author_email='victor.j.forouhar@durham.ac.uk',
      url='https://github.com/VictorForouhar/hdf5Lib',
      packages=['hdf5Lib'],
      install_requires=[
            'h5py',
            'numpy>=1.19.0',
            'tqdm>=4.62.0'
      ],
     )
