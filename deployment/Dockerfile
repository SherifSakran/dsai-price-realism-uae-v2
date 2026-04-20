# SageMaker BYOC (Bring Your Own Container) Dockerfile for Outlier Detection
# Follows SageMaker container conventions

FROM python:3.12.10

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    nginx \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up the program in the image
ENV PATH="/opt/program:${PATH}"

# Set up working directory
WORKDIR /opt/program

# Copy requirements and install Python dependencies
COPY deployment/requirements.txt /opt/program/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deployment/serve.py /opt/program/
COPY deployment/serve.sh /opt/program/
COPY deployment/src/ /opt/program/deployment/src/
COPY deployment/configs/ /opt/program/deployment/configs/

# Make serve.sh executable
RUN chmod +x /opt/program/serve.sh

# Expose port 8080 (SageMaker standard)
EXPOSE 8080

# Use wrapper script as entrypoint to ignore SageMaker's serve argument
ENTRYPOINT ["/opt/program/serve.sh"]
