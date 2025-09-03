"""Schema management for nodetool models.

Provides utility functions to create and drop all database tables defined by the models
in this package. It imports all necessary model classes and maintains a list of them
for batch operations.
"""

from nodetool.config.environment import Environment
import logging

from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.message import Message
from nodetool.models.prediction import Prediction
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow

log = logging.getLogger(__name__)

models = [Asset, Job, Message, Prediction, Thread, Workflow]


async def create_all_tables():
    """Create all database tables for the registered models.

    Iterates through the `models` list and calls the `create_table` class method
    on each model. Logs the creation of each table.
    """
    for model in models:
        await model.create_table()


async def drop_all_tables():
    """Drop all database tables for the registered models.

    Iterates through the `models` list and calls the `drop_table` class method
    on each model. Logs the dropping of each table.
    """
    for model in models:
        await model.drop_table()
