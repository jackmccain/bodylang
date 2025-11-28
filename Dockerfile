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
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Use start.sh as entrypoint to properly handle environment variables
ENTRYPOINT ["/app/start.sh"]
