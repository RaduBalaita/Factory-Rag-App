# Update Dockerfile with correct file paths
FROM node:22.14.0-alpine AS frontend-build

# Build frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Main image with Python backend - using Ubuntu for better compatibility
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy and install Python dependencies with fixed versions
COPY requirements-docker.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend from first stage
COPY --from=frontend-build /app/frontend/build ./frontend/build

# Create necessary directories for persistence
RUN mkdir -p ./docs/PDF ./backend/faiss_index ./.models

# Copy docker configuration files
COPY docker/nginx.conf /etc/nginx/nginx.conf
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

# Create empty .env file in backend directory
RUN touch ./backend/.env

# Create volumes for persistence (this sets the default mount points)
VOLUME ["/app/docs", "/app/backend/faiss_index", "/app/.models"]

# Expose port
EXPOSE 8080

# Start services
CMD ["/start.sh"]