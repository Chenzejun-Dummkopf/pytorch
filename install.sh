#!/bin/sh
conda_path=$(which conda)
if [ -z "$conda_path" ] ; then
  echo "conda is not installed on the machine, install it first please"
  exit 1
else
  echo "conda is in $conda_path"
fi
#python setup.py clean
export NO_CUDA=1
export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
CC=icc CXX=icpc python setup.py install

