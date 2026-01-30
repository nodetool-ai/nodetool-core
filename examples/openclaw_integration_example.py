#!/usr/bin/env python
"""Example script demonstrating OpenClaw integration with nodetool-core.

This script shows how to:
1. Start nodetool-core with OpenClaw enabled
2. Register with the OpenClaw Gateway
3. Query node capabilities and status
4. Execute tasks on the node

Requirements:
- Set OPENCLAW_ENABLED=true
- Set OPENCLAW_GATEWAY_URL (optional, defaults to https://gateway.openclaw.ai)
- Set OPENCLAW_GATEWAY_TOKEN if required by your Gateway
"""

import asyncio
import os
from datetime import datetime

import httpx


async def main():
    """Demonstrate OpenClaw integration."""
    
    # Nodetool API URL (assuming default local server)
    base_url = os.environ.get("NODETOOL_API_URL", "http://localhost:7777")
    openclaw_url = f"{base_url}/openclaw"
    
    print("=" * 60)
    print("OpenClaw Integration Example")
    print("=" * 60)
    print()
    
    async with httpx.AsyncClient() as client:
        # 1. Check health
        print("1. Checking node health...")
        response = await client.get(f"{openclaw_url}/health")
        health = response.json()
        print(f"   Status: {health['status']}")
        print(f"   Node ID: {health['node_id']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f} seconds")
        print()
        
        # 2. Get capabilities
        print("2. Fetching node capabilities...")
        response = await client.get(f"{openclaw_url}/capabilities")
        capabilities = response.json()
        print(f"   Found {len(capabilities)} capabilities:")
        for cap in capabilities:
            print(f"   - {cap['name']}: {cap['description']}")
        print()
        
        # 3. Get detailed status
        print("3. Getting node status...")
        response = await client.get(f"{openclaw_url}/status")
        status = response.json()
        print(f"   Node ID: {status['node_id']}")
        print(f"   Status: {status['status']}")
        print(f"   Active Tasks: {status['active_tasks']}")
        print(f"   Completed: {status['total_tasks_completed']}")
        print(f"   Failed: {status['total_tasks_failed']}")
        
        if "system_info" in status:
            sys_info = status["system_info"]
            print(f"   CPU: {sys_info.get('cpu_percent', 'N/A')}%")
            print(f"   Memory: {sys_info.get('memory_percent', 'N/A')}%")
        print()
        
        # 4. Execute a test task
        print("4. Executing a test task...")
        task_request = {
            "task_id": f"example-task-{datetime.now().timestamp()}",
            "capability_name": "workflow_execution",
            "parameters": {
                "workflow_data": {
                    "nodes": [],
                    "edges": []
                },
                "params": {}
            }
        }
        
        response = await client.post(
            f"{openclaw_url}/execute",
            json=task_request
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Task ID: {result['task_id']}")
            print(f"   Status: {result['status']}")
            print(f"   Message: {result.get('message', 'N/A')}")
            
            # Wait a bit and check task status
            task_id = result['task_id']
            print(f"\n   Waiting for task to complete...")
            await asyncio.sleep(3)
            
            response = await client.get(f"{openclaw_url}/tasks/{task_id}")
            if response.status_code == 200:
                task_status = response.json()
                print(f"   Final status: {task_status['status']}")
                if task_status.get('result'):
                    print(f"   Result available: Yes")
            else:
                print(f"   Task no longer tracked (may have completed)")
        else:
            print(f"   Error: {response.status_code}")
            print(f"   Details: {response.text}")
        print()
        
        # 5. Register with Gateway (if enabled)
        if os.environ.get("OPENCLAW_ENABLED", "").lower() == "true":
            print("5. Registering with OpenClaw Gateway...")
            try:
                response = await client.post(f"{openclaw_url}/register")
                if response.status_code == 200:
                    reg_result = response.json()
                    print(f"   Registration: {'Successful' if reg_result['success'] else 'Failed'}")
                    print(f"   Node ID: {reg_result['node_id']}")
                    if reg_result.get('message'):
                        print(f"   Message: {reg_result['message']}")
                else:
                    print(f"   Registration failed: {response.status_code}")
                    print(f"   Details: {response.text}")
            except Exception as e:
                print(f"   Registration error: {e}")
        else:
            print("5. OpenClaw not enabled (set OPENCLAW_ENABLED=true)")
        
        print()
        print("=" * 60)
        print("Example complete!")
        print("=" * 60)


if __name__ == "__main__":
    print("\nNOTE: This requires nodetool server to be running.")
    print("Start with: OPENCLAW_ENABLED=true nodetool serve\n")
    
    asyncio.run(main())
