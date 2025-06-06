"""Schema management for nodetool models.

Provides utility functions to create and drop all database tables defined by the models
in this package. It imports all necessary model classes and maintains a list of them
for batch operations.
"""

from nodetool.common.environment import Environment

from nodetool.models.asset import Asset
from nodetool.models.job import Job
from nodetool.models.message import Message
from nodetool.models.prediction import Prediction
from nodetool.models.thread import Thread
from nodetool.models.workflow import Workflow

log = Environment.get_logger()

models = [Asset, Job, Message, Prediction, Thread, Workflow]


def create_all_tables():
    """Create all database tables for the registered models.

    Iterates through the `models` list and calls the `create_table` class method
    on each model. Logs the creation of each table.
    """
    for model in models:
        model.create_table()


def drop_all_tables():
    """Drop all database tables for the registered models.

    Iterates through the `models` list and calls the `drop_table` class method
    on each model. Logs the dropping of each table.
    """
    for model in models:
        model.drop_table()
