#!/bin/bash
# Set up comprehensive library path for OpenCV and dependencies
export LD_LIBRARY_PATH="/root/.nix-profile/lib:$LD_LIBRARY_PATH"

# Start the application
exec gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:$PORT --timeout 120 web_app:app
