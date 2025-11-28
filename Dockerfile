FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
ENV PORT=8080
EXPOSE 8080

# Start command (using shell form to allow variable expansion)
CMD ["sh", "-c", "gunicorn --worker-class gevent --workers 1 --bind 0.0.0.0:${PORT} --timeout 120 web_app:app"]
