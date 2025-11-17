import os

from nodetool.api.server import create_app

static_folder = os.getenv("STATIC_FOLDER", "web/dist")

app = create_app(static_folder=static_folder)
