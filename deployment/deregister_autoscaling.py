#!/usr/bin/env python
"""
Deregister auto-scaling targets for a SageMaker endpoint.

This script should be run with dsa-production account credentials
that have permissions for application-autoscaling operations.
"""

import boto3
import os
import sys
from pathlib import Path
from dotenv import load_dotenv


def deregister_autoscaling_targets(
    endpoint_name: str,
    region: str = 'ap-southeast-1'
):
    """
    Deregister any auto-scaling targets for the endpoint.
    
    Args:
        endpoint_name: Name of the endpoint
        region: AWS region
    """
    autoscaling = boto3.client('application-autoscaling', region_name=region)
    
    try:
        # List all scalable targets for SageMaker
        response = autoscaling.describe_scalable_targets(
            ServiceNamespace='sagemaker'
        )
        
        # Find targets for this endpoint
        targets_found = False
        for target in response.get('ScalableTargets', []):
            resource_id = target['ResourceId']
            # Check if this target is for our endpoint
            if f'endpoint/{endpoint_name}/' in resource_id:
                targets_found = True
                print(f"Found auto-scaling target: {resource_id}")
                print(f"  Scalable Dimension: {target['ScalableDimension']}")
                print(f"Deregistering auto-scaling target...")
                
                autoscaling.deregister_scalable_target(
                    ServiceNamespace='sagemaker',
                    ResourceId=resource_id,
                    ScalableDimension=target['ScalableDimension']
                )
                print(f"Auto-scaling target deregistered successfully.")
        
        if not targets_found:
            print(f"No auto-scaling targets found for endpoint {endpoint_name}.")
        else:
            print(f"\nAll auto-scaling targets for {endpoint_name} have been deregistered.")
            
        return True
            
    except Exception as e:
        print(f"Error checking/deregistering auto-scaling targets: {str(e)}")
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
    required_vars = ['ENDPOINT_NAME', 'REGION']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Error: Missing required variables in .env file: {', '.join(missing_vars)}")
        sys.exit(1)
    
    # Build configuration
    config = {
        'endpoint_name': os.getenv('ENDPOINT_NAME'),
        'region': os.getenv('REGION')
    }
    
    return config


def main():
    # Load configuration from .env file
    print("Loading configuration from .env file...")
    config = load_config()
    
    print()
    print("=" * 80)
    print("Deregistering Auto-Scaling Targets for SageMaker Endpoint")
    print("=" * 80)
    print(f"Endpoint Name: {config['endpoint_name']}")
    print(f"Region: {config['region']}")
    print()
    print("NOTE: This script requires dsa-production account credentials")
    print("      with application-autoscaling permissions.")
    print("=" * 80)
    print()
    
    # Deregister auto-scaling targets
    if not deregister_autoscaling_targets(
        endpoint_name=config['endpoint_name'],
        region=config['region']
    ):
        print("\nFailed to deregister auto-scaling targets.")
        return 1
    
    print()
    print("=" * 80)
    print("Auto-Scaling Deregistration Complete!")
    print("=" * 80)
    print()
    print("You can now proceed to delete/update the endpoint using:")
    print("  python deploy_to_sagemaker.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
