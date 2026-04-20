#!/usr/bin/env python
"""
Deploy the outlier detection model to SageMaker.

This script creates a SageMaker model and endpoint for outlier detection.
Reads configuration from .env file in the same directory.

IMPORTANT: If the endpoint has auto-scaling enabled, you must first run
deregister_autoscaling.py with dsa-production credentials before running
this script with TEAM credentials.

Two-step deployment process:
1. Run deregister_autoscaling.py (with dsa-production credentials)
2. Run deploy_to_sagemaker.py (with TEAM credentials)
"""

import boto3
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


def create_model(
    model_name: str,
    image_uri: str,
    model_data_url: str,
    role_arn: str,
    region: str = 'ap-southeast-1'
):
    """
    Create a SageMaker model.
    
    Args:
        model_name: Name for the model
        image_uri: ECR image URI
        model_data_url: S3 URL to model artifacts (lookup table)
        role_arn: IAM role ARN for SageMaker
        region: AWS region
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Creating SageMaker model: {model_name}")
    
    try:
        response = sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'ModelDataUrl': model_data_url,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'serve.py',
                    'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model'
                }
            },
            ExecutionRoleArn=role_arn
        )
        print(f"Model created: {response['ModelArn']}")
        return True
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        return False


def create_endpoint_config(
    config_name: str,
    model_name: str,
    instance_type: str = 'ml.m5.xlarge',
    instance_count: int = 1,
    region: str = 'ap-southeast-1'
):
    """
    Create a SageMaker endpoint configuration.
    
    Args:
        config_name: Name for the endpoint configuration
        model_name: Name of the model to use
        instance_type: EC2 instance type
        instance_count: Number of instances
        region: AWS region
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Creating endpoint configuration: {config_name}")
    
    try:
        response = sagemaker.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': instance_count,
                    'InitialVariantWeight': 1.0
                }
            ]
        )
        print(f"Endpoint config created: {response['EndpointConfigArn']}")
        return True
    except Exception as e:
        print(f"Error creating endpoint config: {str(e)}")
        return False


def delete_endpoint_if_exists(
    endpoint_name: str,
    region: str = 'ap-southeast-1'
):
    """
    Delete a SageMaker endpoint if it exists.
    
    Args:
        endpoint_name: Name of the endpoint to delete
        region: AWS region
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    try:
        # Check if endpoint exists and get its status
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_status = response['EndpointStatus']
        print(f"Endpoint {endpoint_name} exists with status: {endpoint_status}. Deleting...")
        
        # Delete the endpoint
        sagemaker.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint {endpoint_name} deletion initiated.")
        
        # Only wait for deletion if endpoint was not in a Failed state
        # Failed endpoints may not transition properly through the waiter
        if endpoint_status != 'Failed':
            print("Waiting for endpoint deletion to complete...")
            waiter = sagemaker.get_waiter('endpoint_deleted')
            waiter.wait(EndpointName=endpoint_name)
            print("Endpoint deletion complete.")
        else:
            # For failed endpoints, just poll until it's gone
            print("Endpoint was in Failed state. Polling for deletion...")
            import time
            max_attempts = 60
            for attempt in range(max_attempts):
                try:
                    sagemaker.describe_endpoint(EndpointName=endpoint_name)
                    time.sleep(5)
                except sagemaker.exceptions.ClientError as e:
                    if 'Could not find endpoint' in str(e) or 'ValidationException' in str(e):
                        print("Endpoint deletion complete.")
                        break
                    raise
            else:
                print("Warning: Endpoint deletion timed out, but proceeding anyway.")
        
    except sagemaker.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            print(f"Endpoint {endpoint_name} does not exist. Proceeding with creation.")
        else:
            print(f"Error checking/deleting endpoint: {str(e)}")
            raise
    except Exception as e:
        print(f"Error deleting endpoint: {str(e)}")
        raise


def create_endpoint(
    endpoint_name: str,
    config_name: str,
    region: str = 'ap-southeast-1',
    wait: bool = True
):
    """
    Create a SageMaker endpoint.
    
    Args:
        endpoint_name: Name for the endpoint
        config_name: Name of the endpoint configuration to use
        region: AWS region
        wait: Whether to wait for endpoint to be in service
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Creating endpoint: {endpoint_name}")
    
    try:
        response = sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"Endpoint creation initiated: {response['EndpointArn']}")
        
        if wait:
            print("Waiting for endpoint to be in service (this may take several minutes)...")
            waiter = sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(EndpointName=endpoint_name)
            print(f"Endpoint {endpoint_name} is now in service!")
        
        return True
    except Exception as e:
        print(f"Error creating endpoint: {str(e)}")
        return False


def load_config():
    """
    Load configuration from .env file.
    
    Returns:
        dict: Configuration dictionary
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    env_file = script_dir / '.env'
    
    if not env_file.exists():
        print(f"Error: .env file not found at {env_file}")
        print("Please create a .env file with the required configuration.")
        sys.exit(1)
    
    # Load environment variables from .env file
    load_dotenv(env_file)
    
    # Required variables
    required_vars = ['IMAGE_NAME', 'REGION', 'ACCOUNT_ID', 'ROLE_ARN', 'MODEL_DATA_URL', 'ENDPOINT_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required variables in .env file: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Build configuration
    config = {
        'image_name': os.getenv('IMAGE_NAME'),
        'region': os.getenv('REGION'),
        'account_id': os.getenv('ACCOUNT_ID'),
        'role_arn': os.getenv('ROLE_ARN'),
        'model_data_url': os.getenv('MODEL_DATA_URL'),
        'endpoint_name': os.getenv('ENDPOINT_NAME'),
        'instance_type': os.getenv('INSTANCE_TYPE', 'ml.m5.xlarge'),
        'instance_count': int(os.getenv('INSTANCE_COUNT', '1'))
    }
    
    # Build image URI (repo managed by DevOps, image_name used as tag)
    ecr_repo_name = os.getenv('ECR_REPO_NAME', 'price-realism-eg')
    image_tag = config['image_name']
    config['image_uri'] = f"{config['account_id']}.dkr.ecr.{config['region']}.amazonaws.com/{ecr_repo_name}:{image_tag}"
    
    return config


def main():
    # Load configuration from .env file
    print("Loading configuration from .env file...")
    config = load_config()
    
    # Generate unique names with timestamp
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = f"{config['endpoint_name']}-model-{timestamp}"
    config_name = f"{config['endpoint_name']}-config-{timestamp}"
    
    print()
    print("=" * 80)
    print("Deploying Outlier Detection Model to SageMaker")
    print("=" * 80)
    print(f"Endpoint Name: {config['endpoint_name']}")
    print(f"Model Name: {model_name}")
    print(f"Config Name: {config_name}")
    print(f"Image URI: {config['image_uri']}")
    print(f"Model Data: {config['model_data_url']}")
    print(f"Role ARN: {config['role_arn']}")
    print(f"Instance Type: {config['instance_type']}")
    print(f"Instance Count: {config['instance_count']}")
    print(f"Region: {config['region']}")
    print("=" * 80)
    print()
    
    # Step 1: Create model
    if not create_model(
        model_name=model_name,
        image_uri=config['image_uri'],
        model_data_url=config['model_data_url'],
        role_arn=config['role_arn'],
        region=config['region']
    ):
        print("Failed to create model. Exiting.")
        return 1
    
    print()
    
    # Step 2: Create endpoint configuration
    if not create_endpoint_config(
        config_name=config_name,
        model_name=model_name,
        instance_type=config['instance_type'],
        instance_count=config['instance_count'],
        region=config['region']
    ):
        print("Failed to create endpoint configuration. Exiting.")
        return 1
    
    print()
    
    # Step 3: Delete existing endpoint if it exists
    try:
        delete_endpoint_if_exists(
            endpoint_name=config['endpoint_name'],
            region=config['region']
        )
    except Exception as e:
        print(f"Failed to delete existing endpoint: {str(e)}")
        return 1
    
    print()
    
    # Step 4: Create endpoint
    if not create_endpoint(
        endpoint_name=config['endpoint_name'],
        config_name=config_name,
        region=config['region'],
        wait=True
    ):
        print("Failed to create endpoint. Exiting.")
        return 1
    
    print()
    print("=" * 80)
    print("Deployment Complete!")
    print("=" * 80)
    print(f"Endpoint Name: {config['endpoint_name']}")
    print(f"Region: {config['region']}")
    print()
    print("You can now invoke the endpoint using:")
    print(f"  aws sagemaker-runtime invoke-endpoint \\")
    print(f"    --endpoint-name {config['endpoint_name']} \\")
    print(f"    --region {config['region']} \\")
    print(f"    --body file://sample_request.json \\")
    print(f"    --content-type application/json \\")
    print(f"    response.json")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
