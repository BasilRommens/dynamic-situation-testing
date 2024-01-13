#!/bin/bash

# create a venv
python3 -m venv venv

# activate the venv
source venv/bin/activate

# upgrade pip
pip3 install --upgrade pip

# Install dependencies
pip3 install -r requirements.txt

# Setup the distance function written in cython
cd src
python3 cython_setup.py build_ext --inplace
