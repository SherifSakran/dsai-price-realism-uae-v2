#!/usr/bin/env python
"""
Deploy the outlier detection model to SageMaker.

This script creates or updates a SageMaker model and endpoint for outlier detection.
Reads configuration from .env file in the same directory.

IMPORTANT: If the endpoint has auto-scaling enabled, you must first run
deregister_autoscaling.py with dsa-production credentials before running
this script with TEAM credentials. After deployment, re-register autoscaling
by running register_autoscaling.py with dsa-production credentials.

Zero-downtime deployment process:
1. Run deregister_autoscaling.py (with dsa-production credentials)
2. Run deploy_to_sagemaker.py (with TEAM credentials) — uses update_endpoint for zero downtime
3. Run register_autoscaling.py (with dsa-production credentials)
"""

import boto3
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv


def create_model(
    model_name: str,
    image_uri: str,
    role_arn: str,
    region: str = 'ap-southeast-1',
    lookup_s3_bucket: str = 'dsai-price-realism-staging',
    lookup_s3_key: str = 'uae/lookup/segment_lookup_table.parquet',
    location_tree_s3_key: str = 'uae/lookup/location_tree_lookup.parquet',
):
    """
    Create a SageMaker model.
    
    Args:
        model_name: Name for the model
        image_uri: ECR image URI
        role_arn: IAM role ARN for SageMaker
        region: AWS region
        lookup_s3_bucket: S3 bucket for lookup table and feedback
        lookup_s3_key: S3 key for the segment lookup table parquet
        location_tree_s3_key: S3 key for the location tree lookup parquet
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Creating SageMaker model: {model_name}")
    print(f"Setting LOOKUP_TABLE_S3_BUCKET={lookup_s3_bucket}")
    print(f"Setting LOOKUP_TABLE_S3_KEY={lookup_s3_key}")
    print(f"Setting LOCATION_TREE_S3_KEY={location_tree_s3_key}")
    
    try:
        response = sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'Image': image_uri,
                'Environment': {
                    'SAGEMAKER_PROGRAM': 'serve.py',
                    'LOOKUP_TABLE_S3_BUCKET': lookup_s3_bucket,
                    'LOOKUP_TABLE_S3_KEY': lookup_s3_key,
                    'LOCATION_TREE_S3_KEY': location_tree_s3_key,
                    'CX_FEEDBACK_S3_BUCKET': lookup_s3_bucket,
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


def endpoint_exists(
    endpoint_name: str,
    region: str = 'ap-southeast-1'
):
    """
    Check if a SageMaker endpoint exists and return its status.
    
    Args:
        endpoint_name: Name of the endpoint
        region: AWS region
        
    Returns:
        str or None: Endpoint status if it exists, None otherwise
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        return response['EndpointStatus']
    except sagemaker.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            return None
        raise


def update_endpoint(
    endpoint_name: str,
    config_name: str,
    region: str = 'ap-southeast-1',
    wait: bool = True
):
    """
    Update an existing SageMaker endpoint with a new configuration.
    This performs a zero-downtime rolling (blue/green) update.
    
    Args:
        endpoint_name: Name of the endpoint to update
        config_name: Name of the new endpoint configuration
        region: AWS region
        wait: Whether to wait for the update to complete
        
    Returns:
        tuple: (success: bool, error: Exception or None)
    """
    sagemaker = boto3.client('sagemaker', region_name=region)
    
    print(f"Updating endpoint: {endpoint_name} with config: {config_name}")
    print("This performs a zero-downtime rolling update (blue/green deployment).")
    
    try:
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"Endpoint update initiated.")
        
        if wait:
            print("Waiting for endpoint update to complete (this may take several minutes)...")
            waiter = sagemaker.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={'Delay': 30, 'MaxAttempts': 60}
            )
            print(f"Endpoint {endpoint_name} update complete and in service!")
        
        return True, None
    except Exception as e:
        print(f"Error updating endpoint: {str(e)}")
        return False, e


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
    required_vars = ['IMAGE_NAME', 'REGION', 'ACCOUNT_ID', 'ROLE_ARN', 'ENDPOINT_NAME']
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
        'endpoint_name': os.getenv('ENDPOINT_NAME'),
        'instance_type': os.getenv('INSTANCE_TYPE', 'ml.m5.xlarge'),
        'instance_count': int(os.getenv('INSTANCE_COUNT', '1')),
        'lookup_s3_bucket': os.getenv('LOOKUP_TABLE_S3_BUCKET', 'dsai-price-realism-staging'),
        'lookup_s3_key': os.getenv('LOOKUP_TABLE_S3_KEY', 'uae/lookup/segment_lookup_table.parquet'),
        'location_tree_s3_key': os.getenv('LOCATION_TREE_S3_KEY', 'uae/lookup/location_tree_lookup.parquet'),
    }
    
    # Build image URI (repo managed by DevOps, image_name used as tag)
    ecr_repo_name = os.getenv('ECR_REPO_NAME', 'price-realism-eg')
    image_tag = config['image_name']
    config['image_uri'] = f"{config['account_id']}.dkr.ecr.{config['region']}.amazonaws.com/{ecr_repo_name}:{image_tag}"
    
    return config


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Deploy the outlier detection model to SageMaker',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT: If the endpoint has auto-scaling enabled, you must first run
deregister_autoscaling.py with dsa-production credentials before running
this script with TEAM credentials. After deployment, re-register autoscaling
by running register_autoscaling.py with dsa-production credentials.

Zero-downtime deployment process:
1. Run deregister_autoscaling.py (with dsa-production credentials)
2. Run deploy_to_sagemaker.py (with TEAM credentials)
3. Run register_autoscaling.py (with dsa-production credentials)

If you don't have permissions to update the endpoint, use --force-deployment:
  python deploy_to_sagemaker.py --force-deployment
        """
    )
    parser.add_argument(
        '--force-deployment',
        action='store_true',
        help='Force deployment by deleting and recreating the endpoint if update fails due to permissions'
    )
    args = parser.parse_args()
    
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
    print(f"Lookup S3: s3://{config['lookup_s3_bucket']}/{config['lookup_s3_key']}")
    print(f"Location Tree S3: s3://{config['lookup_s3_bucket']}/{config['location_tree_s3_key']}")
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
        role_arn=config['role_arn'],
        region=config['region'],
        lookup_s3_bucket=config['lookup_s3_bucket'],
        lookup_s3_key=config['lookup_s3_key'],
        location_tree_s3_key=config['location_tree_s3_key'],
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
    
    # Step 3: Update or create endpoint
    status = endpoint_exists(
        endpoint_name=config['endpoint_name'],
        region=config['region']
    )
    
    if status is not None:
        print(f"Endpoint {config['endpoint_name']} exists with status: {status}")
        print("Performing zero-downtime update...")
        print()
        success, error = update_endpoint(
            endpoint_name=config['endpoint_name'],
            config_name=config_name,
            region=config['region'],
            wait=True
        )
        
        if not success:
            # Check if it's an AccessDeniedException
            is_access_denied = error and 'AccessDeniedException' in str(type(error).__name__)
            if not is_access_denied:
                # Try to check the error message as well
                is_access_denied = error and 'AccessDeniedException' in str(error)
            
            if is_access_denied:
                print()
                print("=" * 80)
                print("ACCESS DENIED: You don't have permission to update the endpoint.")
                print("=" * 80)
                
                if args.force_deployment:
                    print("--force-deployment flag is set. Deleting and recreating the endpoint...")
                    print("WARNING: This will cause downtime during the deployment.")
                    print()
                    
                    # Delete the existing endpoint
                    try:
                        delete_endpoint_if_exists(
                            endpoint_name=config['endpoint_name'],
                            region=config['region']
                        )
                    except Exception as e:
                        print(f"Failed to delete existing endpoint: {str(e)}")
                        return 1
                    
                    print()
                    
                    # Create the new endpoint
                    if not create_endpoint(
                        endpoint_name=config['endpoint_name'],
                        config_name=config_name,
                        region=config['region'],
                        wait=True
                    ):
                        print("Failed to create endpoint. Exiting.")
                        return 1
                else:
                    print("To force deployment by deleting and recreating the endpoint, run:")
                    print(f"  python {sys.argv[0]} --force-deployment")
                    print()
                    print("WARNING: Using --force-deployment will cause downtime during deployment.")
                    return 1
            else:
                print("Failed to update endpoint. Exiting.")
                return 1
    else:
        print(f"Endpoint {config['endpoint_name']} does not exist. Creating new endpoint...")
        print()
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
    print("Next step: Re-register auto-scaling with dsa-production credentials:")
    print(f"  python register_autoscaling.py")
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
