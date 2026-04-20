# Price Realism Outlier Detection - SageMaker Deployment

This directory contains the SageMaker inference endpoint for outlier detection using lookup tables.

## Overview

The outlier detection service uses pre-computed segment statistics (lookup table) to identify price outliers in real-time. It follows the same segmentation and multiplier logic as the batch inference script but is optimized for real-time API requests.

## Files

### Core Application
- **`serve.py`**: FastAPI-based inference server implementing `/ping` and `/invocations` endpoints
- **`src/`**: Source code modules
  - `schemas.py`: Pydantic request/response models and enums
  - `model_loader.py`: Lookup table and artifact loading
  - `inference.py`: Main request processing logic
  - `segmentation.py`: Segment key creation and assignment
  - `outlier_detection.py`: Multiplier determination and outlier checking
  - `feedback_loop.py`: CX feedback loop implementation
  - `utils.py`: Utility functions (normalization, formatting)
- **`configs/`**: Configuration files
  - `constants.py`: Core constants (property types, multipliers, thresholds)
  - `feedback_config.py`: CX feedback loop configuration

### Deployment
- **`Dockerfile`**: Docker container definition for SageMaker BYOC (Bring Your Own Container)
- **`requirements.txt`**: Python dependencies
- **`serve.sh`**: Entrypoint script for the container
- **`build_and_push.sh`**: Script to build and push Docker image to ECR
- **`deploy_to_sagemaker.py`**: Script to deploy the model to SageMaker

### Testing
- **`test_local.py`**: Local testing script
- **`test_endpoint.py`**: SageMaker endpoint testing
- **`compare_endpoint_vs_inference.py`**: Compare endpoint vs batch inference results

## Prerequisites

1. AWS CLI configured with appropriate credentials
2. Docker installed and running
3. IAM role with SageMaker permissions
4. Lookup table file (`segment_lookup_table.parquet`) uploaded to S3

## Deployment Steps

### 1. Configure Environment Variables

Create a `.env` file in the `deployment` directory:

```bash
cd deployment
cp .env.example .env
```

Edit `.env` with your configuration:

```bash
# Docker Image Configuration
IMAGE_NAME=price-realism-outlier-detection
REGION=ap-southeast-1
# ACCOUNT_ID will be auto-detected if not provided

# AWS Credentials (Optional - uses AWS CLI default profile if not set)
# AWS_ACCESS_KEY_ID=your-access-key-id
# AWS_SECRET_ACCESS_KEY=your-secret-access-key
# AWS_PROFILE=default
```

**Note:** AWS credentials are optional in the `.env` file. If not provided, the script will use your AWS CLI default profile or environment variables already set in your shell.

### 2. Prepare Model Artifacts

Create a model.tar.gz file:

```bash
cd ../lookup_tables
tar -czf model.tar.gz segment_lookup_table_<version>.parquet
aws s3 cp model.tar.gz s3://dsai-price-realism-staging/uae/sm/model.tar.gz
```
- s3://dsai-price-realism-staging/uae/sm/model.tar.gz
- s3://dsai-price-realism-production/uae/sm/model.tar.gz

To upload a lookup or feedback table to S3:

```bash
aws s3 cp ../lookup_tables/segment_lookup_table_<version>.parquet \
  s3://your-bucket/price-realism-outlier-detection/segment_lookup_table.parquet
```
### 3. Build and Push Docker Image

```bash
cd deployment
chmod +x build_and_push.sh
./build_and_push.sh
```

This will:
- Build the Docker image
- Create an ECR repository (if it doesn't exist)
- Push the image to ECR

### 4. Deploy to SageMaker

The deployment script reads all configuration from the `.env` file:

```bash
python deploy_to_sagemaker.py
```

The script will:
- Load configuration from `.env`
- Create a SageMaker model
- Create an endpoint configuration
- Deploy the endpoint
- Wait for the endpoint to be in service

Make sure your `.env` file has all required SageMaker configuration variables set.

## Local Testing

### 1. Setup Local Test Environment

Run the setup script to prepare the local model directory:

```bash
cd deployment
chmod +x setup_local_test.sh
./setup_local_test.sh
```

This will:
- Create a `local_model/` directory
- Copy the latest lookup table to `local_model/segment_lookup_table.parquet`

### 2. Run the Server Locally

```bash
MODEL_PATH=./local_model python serve.py
```

The server will start on `http://localhost:8080`

### 3. Test with the Test Script

In a new terminal, run:

```bash
cd deployment
python test_local.py
```

### 4. Manual Testing with curl

Health check:

```bash
curl http://localhost:8080/ping
```

Single request:

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "property_type": "Apartment",
    "location_id": "2572",
    "price_type": "sale",
    "category": "residential",
    "property_sqft": 1200,
    "bedrooms": "2",
    "price": 1500000
  }'
```

Batch request:

```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '[
    {
      "property_type": "Apartment",
      "location_id": "2572",
      "price_type": "sale",
      "category": "residential",
      "property_sqft": 1200,
      "bedrooms": "2",
      "price": 1500000
    },
    {
      "property_type": "Villa",
      "location_id": "2572",
      "price_type": "yearly",
      "category": "residential",
      "property_sqft": 3000,
      "bedrooms": "4",
      "price": 150000
    }
  ]'
```

## API Contract Summary

### Request Schema

```json
{
  "property_type": "Apartment",
  "location_id": "2572",
  "price_type": "sale",
  "category": "residential",
  "property_sqft": 1200,
  "bedrooms": "2",
  "bathrooms": "2",
  "floor_number": "5",
  "completion_status": "completed",
  "furnishing_type": "unfurnished",
  "property_parking": "2",
  "price": 1500000
}
```

**Required Fields:**
- `property_type`: Property type (see enum in `serve.py`)
- `location_id`: Location ID (string)
- `price_type`: One of `sale`, `monthly`, `yearly`, `weekly`, `daily`
- `category`: One of `residential`, `commercial`
- `property_sqft`: Property size in sqft (float > 0)
- `price`: Price to check for outliers (float > 0)

**Optional Fields:**
- `bedrooms`: Number of bedrooms or "studio" (string)
- `bathrooms`: Number of bathrooms (string)
- `floor_number`: Floor number (string)
- `completion_status`: One of `completed`, `off_plan`, `completed_primary`, `off_plan_primary`
- `furnishing_type`: One of `furnished`, `semi-furnished`, `unfurnished`
- `property_parking`: Parking count (string)
- `property_listing_id`: Listing ID for feedback lookup (string)

### Response Schema

```json
{
  "valid_size": true,
  "unseen": false,
  "lower_bound": 850000.0,
  "upper_bound": 3600000.0,
  "segment_abs_pct_error": null
}
```

**Response Fields:**
- `valid_size`: Boolean indicating if the property size is valid (not flagged as both sqft outlier AND price-to-sqft outlier)
- `unseen`: Boolean indicating whether the segment was found in the lookup table (true if segment not found for not having sufficient supply)
- `lower_bound`: Lower price bound based on price-to-sqft median and multiplier (null if unseen or invalid size)
- `upper_bound`: Upper price bound based on price-to-sqft median and multiplier (null if unseen or invalid size)
- `segment_abs_pct_error`: Reserved field (currently always null)

### Error Response

```json
{
  "error": "Validation error message"
}
```

## Outlier Detection Logic

The service follows the same logic as the batch inference script:

1. **Segment Assignment**: Assigns listings to segments using the lookup table
   - For Top Residential Types: Try with bedrooms first, then without
   - For Rest of Housing Types: Try location hierarchy (location_id → lvl_3 → lvl_2 → lvl_1 → lvl_0) if rollup is enabled

2. **Multiplier Selection**:
   - **Top Residential Types**: 
     - Use 4x multiplier when EITHER condition is met:
       - `segment_count >= 20` (large segment = more confidence) OR
       - `relaxation_level == 'without_bedrooms'` (already relaxed segmentation)
     - Otherwise use 3x multiplier
   - **Rest of Housing Types**: Use location-based multipliers (6x to 10x) if rollup is enabled

3. **Outlier Detection**:
   - High outlier: `value > multiplier * median`
   - Low outlier: `value < (1 / multiplier) * median`
   - Prioritise price_to_sqft: price bounds are computed using price_to_sqft and size_outliers are only reported iff they are price_to_sqft outliers as well.
