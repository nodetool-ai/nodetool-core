"""
FAL.ai pricing API integration.

This module provides functionality to fetch pricing information from FAL.ai's
pricing API endpoint. FAL uses output-based pricing (e.g., per image/video)
with proportional adjustments for resolution and length.

FAL API Documentation: https://docs.fal.ai
"""

from __future__ import annotations

import aiohttp

from nodetool.config.logging_config import get_logger
from nodetool.metadata.types import ModelPricing, Provider

log = get_logger(__name__)

FAL_API_BASE_URL = "https://api.fal.ai/v1"


async def fetch_fal_pricing(
    api_key: str,
    endpoint_ids: list[str] | None = None,
) -> list[ModelPricing]:
    """Fetch pricing information from FAL.ai's pricing API.

    Args:
        api_key: FAL API key for authentication.
        endpoint_ids: Optional list of specific endpoint IDs to get pricing for.
                     If None, returns pricing for all available endpoints.
                     Accepts 1-50 endpoint IDs.

    Returns:
        List of ModelPricing instances with pricing information.
        Returns empty list if pricing information is not available.
    """
    if not api_key:
        log.debug("No FAL API key provided, returning empty pricing list")
        return []

    if not endpoint_ids:
        log.debug("No endpoint IDs provided for FAL pricing, returning empty list")
        return []

    # FAL requires specific endpoint IDs to be requested
    # Limit to 50 as per API specification
    if len(endpoint_ids) > 50:
        log.warning("FAL pricing API accepts max 50 endpoint IDs, truncating")
        endpoint_ids = endpoint_ids[:50]

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            "Authorization": f"Key {api_key}",
            "Content-Type": "application/json",
        }

        # Build query parameters for endpoint IDs
        params = [("endpoint_id", eid) for eid in endpoint_ids]

        async with aiohttp.ClientSession(timeout=timeout) as session:
            url = f"{FAL_API_BASE_URL}/models/pricing"
            async with session.get(url, headers=headers, params=params) as response:
                if response.status == 401:
                    log.warning("FAL API authentication failed")
                    return []
                if response.status == 400:
                    log.warning("FAL API bad request - invalid endpoint IDs")
                    return []
                if response.status != 200:
                    log.warning(f"Failed to fetch FAL pricing: HTTP {response.status}")
                    return []

                data = await response.json()
                prices = data.get("prices", [])

                pricing_list: list[ModelPricing] = []
                for price_entry in prices:
                    endpoint_id = price_entry.get("endpoint_id", "")
                    unit_price = price_entry.get("unit_price", 0)
                    unit = price_entry.get("unit", "request")
                    currency = price_entry.get("currency", "USD")

                    # Convert price to float if it's a string or int
                    try:
                        unit_price = float(unit_price)
                    except (ValueError, TypeError):
                        unit_price = 0.0

                    pricing_list.append(
                        ModelPricing(
                            endpoint_id=endpoint_id,
                            provider=Provider.HuggingFaceFalAI,
                            unit_price=unit_price,
                            unit=unit,
                            currency=currency,
                        )
                    )

                log.debug(f"Fetched {len(pricing_list)} pricing entries from FAL.ai")
                return pricing_list

    except aiohttp.ClientError as e:
        log.error(f"Network error fetching FAL pricing: {e}")
        return []
    except Exception as e:
        log.error(f"Error fetching FAL pricing: {e}")
        return []
