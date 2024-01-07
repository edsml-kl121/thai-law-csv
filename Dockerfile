# Use a slim Python image
FROM python:3.11-slim as base

# Install build tools and other utilities, then clean up in one layer
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files for installing dependencies
COPY requirements_venv.txt .

# Install any necessary packages specified in req.txt
RUN pip install --no-cache-dir -r requirements_venv.txt

# Set working directory
WORKDIR /usr/src/app

# Copy the application files
COPY . /usr/src/app

RUN touch .env

# Expose port
EXPOSE 8001

CMD ["python", "app.py"]