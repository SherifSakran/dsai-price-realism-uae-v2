#!/bin/bash

# Build and push Docker image to ECR for SageMaker deployment
# Usage: ./build_and_push.sh

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load environment variables from .env file if it exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source "${SCRIPT_DIR}/.env"
    set +a
else
    echo "Warning: .env file not found in ${SCRIPT_DIR}"
    echo "Please create a .env file with IMAGE_NAME, REGION, and optionally ACCOUNT_ID"
    exit 1
fi

# Use environment variables with defaults
IMAGE_NAME=${IMAGE_NAME:-price-realism-outlier-detection}
REGION=${REGION:-ap-southeast-1}
ECR_REPO_NAME=${ECR_REPO_NAME:-price-realism-eg}

# Get AWS account ID (use from env if provided, otherwise fetch)
if [ -z "$ACCOUNT_ID" ]; then
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
fi

# ECR registry and full image URI
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO_NAME}:${IMAGE_NAME}"

echo "Building Docker image: ${IMAGE_NAME}"
echo "ECR Image URI: ${ECR_IMAGE}"

# Build Docker image (from project root)
cd "${SCRIPT_DIR}/.."
# Build for linux/amd64 (SageMaker requirement)
docker buildx build --platform linux/amd64 --provenance=false --output type=docker -t ${IMAGE_NAME}:latest -f deployment/Dockerfile .

# Tag image for ECR
docker tag ${IMAGE_NAME}:latest ${ECR_IMAGE}

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}

# Verify ECR repository exists (non-fatal — cross-account may lack describe permission)
echo "Verifying ECR repository exists..."
aws ecr describe-repositories --repository-names ${ECR_REPO_NAME} --registry-id ${ACCOUNT_ID} --region ${REGION} 2>/dev/null || \
    echo "Warning: Could not verify ECR repository '${ECR_REPO_NAME}' (may lack ecr:DescribeRepositories permission). Attempting push anyway..."

# Push image to ECR
echo "Pushing image to ECR..."
docker push ${ECR_IMAGE}

echo ""
echo "Done! Image pushed to: ${ECR_IMAGE}"
echo "Image is single-platform linux/amd64 with Docker V2 manifest (SageMaker compatible)."
echo ""
echo "Next steps:"
echo "1. Upload your artifacts to S3 as 'model.tar.gz'"
echo "2. Run: python deploy_to_sagemaker.py"
echo "3. Test with: python test_endpoint_simple.py"
