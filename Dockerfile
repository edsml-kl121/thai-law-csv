# Use an official Python runtime as a parent image
FROM  icr.io/client-engineering/python-build-base-itmx-backend:3.11-slim

# Set working directory
WORKDIR /usr/src/app

# Copy the application files
COPY . /usr/src/app

RUN touch .env

# Expose port
EXPOSE 8001

CMD ["python", "app.py"]
