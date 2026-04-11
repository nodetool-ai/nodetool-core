"""
Hugging Face Authentication Module

Retrieves HF tokens from environment variables or secrets.
"""

import os

from nodetool.config.logging_config import get_logger
from nodetool.security.secret_helper import get_secret

log = get_logger(__name__)


async def get_hf_token(user_id: str | None = None) -> str | None:
    if user_id:
        try:
            token = await get_secret("HF_TOKEN", user_id)
            if token:
                return token
        except Exception as e:
            log.debug(f"get_hf_token: Error getting HF_TOKEN secret: {e}")

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    return None
