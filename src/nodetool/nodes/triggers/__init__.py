"""
Trigger Nodes for Infinite Running Workflows
=============================================

This module provides trigger nodes that enable workflows to run indefinitely,
waiting for external events such as webhooks, file system changes, scheduled
intervals, or manual inputs.

Trigger nodes are special streaming nodes that:
1. Block until an external event occurs
2. Emit the event data as output
3. Loop back to wait for the next event
4. Only terminate when explicitly stopped

Core Components:
---------------
- TriggerNode: Base class for all trigger nodes
- WebhookTrigger: Receives HTTP webhook requests
- FileWatchTrigger: Monitors filesystem changes
- IntervalTrigger: Fires at configured time intervals
- ManualTrigger: Waits for manual input via API

Usage:
------
Connect a trigger node as the input to a workflow. The workflow will run
indefinitely, processing each event as it arrives and returning to wait
for the next event.

Example workflow:
    WebhookTrigger -> ProcessData -> OutputNode

This workflow will:
1. Start and wait for webhook events
2. When a webhook arrives, process the data
3. Output the result
4. Return to waiting for the next webhook
"""

from nodetool.nodes.triggers.base import TriggerNode
from nodetool.nodes.triggers.webhook import WebhookTrigger
from nodetool.nodes.triggers.file_watch import FileWatchTrigger
from nodetool.nodes.triggers.interval import IntervalTrigger
from nodetool.nodes.triggers.manual import ManualTrigger

__all__ = [
    "TriggerNode",
    "WebhookTrigger",
    "FileWatchTrigger",
    "IntervalTrigger",
    "ManualTrigger",
]
