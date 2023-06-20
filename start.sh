#!/bin/bash

# enter the venv
source venv/bin/activate

# start the server
python3 src/wsgi.py

# open the default browser using xdg-open
xdg-open http://localhost:5000
