#!/bin/bash

# Install dependencies
pip3 install -r requirements.txt

# Setup the distance function written in cython
cd src
python3 cython_setup.py build_ext --inplace
