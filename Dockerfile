# Use official Python image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy src folder contents into /app
COPY src/ /app/

# Copy requirements.txt if you have one (adjust path if needed)
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app uses (adjust as needed)
EXPOSE 80

# Default command: open a bash shell (container won't run your app automatically)
CMD ["bash"]