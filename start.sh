#!/bin/bash
set -e

# Debug: Print environment info
echo "PORT environment variable: '${PORT}'"
echo "All env vars with PORT:"
env | grep -i port || echo "No PORT vars found"

# Default to 8080 if PORT not set
PORT=${PORT:-8080}
echo "Using PORT: ${PORT}"

# Start the application
exec gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:${PORT} --timeout 120 web_app:app
