#!/usr/bin/env python3
"""
RunPod Workflow Test Script

This script tests deployed NodeTool workflows on RunPod serverless infrastructure.
It sends workflow parameters, monitors execution, and retrieves results.

Based on RunPod serverless API documentation:
https://docs.runpod.io/tutorials/serverless/run-your-first

Usage:
    python test_runpod_workflow.py --endpoint-id YOUR_ENDPOINT_ID --api-key YOUR_API_KEY

Requirements:
    - RunPod API key
    - Deployed RunPod endpoint ID
    - runpod library

Environment Variables:
    RUNPOD_API_KEY: Your RunPod API key (can also be passed as --api-key)
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, Any, Optional
from datetime import datetime

import dotenv
dotenv.load_dotenv()

try:
    import runpod
except ImportError:
    print("❌ Error: runpod library not found")
    print("Install it with: pip install runpod")
    sys.exit(1)

class RunPodWorkflowTester:
    """Test deployed NodeTool workflows on RunPod serverless endpoints."""
    
    def __init__(self, endpoint_id: str, api_key: str):
        """
        Initialize the tester with endpoint ID and API key.
        
        Args:
            endpoint_id (str): RunPod endpoint ID
            api_key (str): RunPod API key
        """
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        
        # Configure runpod library
        runpod.api_key = api_key
        self.endpoint = runpod.Endpoint(endpoint_id)
        
    def run_workflow_sync(self, workflow_params: Dict[str, Any], timeout: int = 60) -> Dict[str, Any]:
        """
        Run a workflow synchronously on RunPod.
        
        Args:
            workflow_params (Dict[str, Any]): Parameters to pass to the workflow
            timeout (int): Timeout in seconds
            
        Returns:
            Dict[str, Any]: Job result from RunPod
            
        Raises:
            TimeoutError: If the job times out
            Exception: If the request fails
        """
        print(f"🚀 Starting workflow execution...")
        print(f"Endpoint ID: {self.endpoint_id}")
        print(f"Parameters: {json.dumps(workflow_params, indent=2)}")
        print(f"Timeout: {timeout} seconds")
        
        try:
            job = self.endpoint.run(workflow_params)
            
            print(f"Job status: {job.status()}")
            while job.status() in ("RUNNING", "IN_PROGRESS", "IN_QUEUE"):
                time.sleep(1)
                print(f"Job status: {job.status()}")

            result = job.output()
            
            print(f"✅ Job completed successfully!")
            print(f"Execution completed in {timeout} seconds or less")
            
            return result
            
        except TimeoutError:
            print(f"⏰ Job timed out after {timeout} seconds")
            raise
        except Exception as e:
            print(f"❌ Failed to run workflow: {e}")
            raise
    
    
    def save_results(self, job_result: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """
        Save job results to a file.
        
        Args:
            job_result (Dict[str, Any]): Job result from RunPod
            output_file (str, optional): Output file path
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"runpod_result_{timestamp}.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(job_result, f, indent=2)
            
            print(f"💾 Results saved to: {output_file}")
            
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    

def load_test_parameters(params_file: str) -> Dict[str, Any]:
    """
    Load test parameters from a JSON file.
    
    Args:
        params_file (str): Path to JSON file with test parameters
        
    Returns:
        Dict[str, Any]: Test parameters
    """
    try:
        with open(params_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load parameters from {params_file}: {e}")
        sys.exit(1)

def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test deployed NodeTool workflow on RunPod")
    parser.add_argument("--endpoint-id", required=True, help="RunPod endpoint ID")
    parser.add_argument("--api-key", help="RunPod API key (can also use RUNPOD_API_KEY env var)")
    parser.add_argument("--params", help="JSON file with workflow parameters")
    parser.add_argument("--params-json", help="Inline JSON string with workflow parameters")
    parser.add_argument("--output", help="Output file for results (default: auto-generated)")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in seconds (default: 60)")
    
    args = parser.parse_args()
    
    # Get API key from argument or environment
    api_key = args.api_key or os.getenv("RUNPOD_API_KEY")
    if not api_key:
        print("❌ Error: RunPod API key is required")
        print("Provide it via --api-key argument or RUNPOD_API_KEY environment variable")
        sys.exit(1)
    
    # Create tester instance
    tester = RunPodWorkflowTester(args.endpoint_id, api_key)
    
    # Get workflow parameters
    if args.params:
        workflow_params = load_test_parameters(args.params)
    elif args.params_json:
        try:
            workflow_params = json.loads(args.params_json)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in --params-json: {e}")
            sys.exit(1)
    else:
        # Use empty parameters as default
        workflow_params = {}
        print("⚠️ No parameters provided, using empty parameters")
    
    print(f"🧪 Testing RunPod workflow...")
    print(f"Endpoint ID: {args.endpoint_id}")
    print(f"Timeout: {args.timeout} seconds")
    
    try:
        # Use synchronous execution
        result = tester.run_workflow_sync(workflow_params, args.timeout)
        
        # Display results
        print(result)
        
        # Save results
        tester.save_results(result, args.output)
        
        print(f"\n✅ Test completed successfully!")
        
    except TimeoutError:
        print(f"\n⏰ Job timed out")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 