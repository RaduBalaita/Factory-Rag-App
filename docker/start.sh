#!/bin/sh

# Print startup info
echo "🐳 Starting RAG Application Docker Container"
echo "📂 Working directory: $(pwd)"
echo "📁 Backend directory: $(ls -la backend/ | head -5)"
echo "🔧 Environment variables:"
echo "   GEMINI_API_KEY: ${GEMINI_API_KEY:-'Not set (will use UI configuration)'}"

# Start supervisor to manage all services
echo "🚀 Starting services with supervisor..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf
