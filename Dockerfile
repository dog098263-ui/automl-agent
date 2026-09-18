FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer cache)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser binaries for scraping
RUN playwright install chromium --with-deps

# Copy backend source
COPY backend/ ./backend/

# Copy frontend source
COPY frontend/ ./frontend/

# Set working directory to backend (where main.py lives)
WORKDIR /app/backend

# Create data directory
RUN mkdir -p data/users/user_1

# Expose port (Render/Railway inject $PORT; default 8000)
EXPOSE 8000

# Start server — reads PORT env var
CMD ["sh", "-c", "python main.py"]
