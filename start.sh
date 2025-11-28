#!/bin/bash
set -e

# Start the application with PORT from Railway environment
exec gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:${PORT} --timeout 120 web_app:app
