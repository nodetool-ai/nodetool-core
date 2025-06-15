from httpx import AsyncClient, HTTPStatusError, RequestError
from nodetool.agents.serp_providers.serp_providers import ErrorResponse, SerpProvider
from nodetool.agents.tools._remove_base64_images import _remove_base64_images
from nodetool.common.environment import Environment


import json
from typing import Any, Dict, List, Union


class SerpApiProvider(SerpProvider):
    """
    A SERP provider that uses the SerpApi.com API.
    API Documentation: https://serpapi.com/search-api
    """

    BASE_URL = "https://serpapi.com/search.json"
    DEFAULT_GL = "us"  # Country
    DEFAULT_HL = "en"  # Language

    def __init__(
        self,
        api_key: str | None = None,
        gl: str = DEFAULT_GL,
        hl: str = DEFAULT_HL,
    ):
        self.api_key = api_key or Environment.get("SERPAPI_API_KEY")
        self.gl = gl
        self.hl = hl
        self._client = AsyncClient(timeout=60.0)

        if not self.api_key:
            raise ValueError(
                "SerpApi API key (SERPAPI_API_KEY) not found or not provided."
            )

    async def _make_request(
        self, params: Dict[str, Any]
    ) -> Union[Dict[str, Any], ErrorResponse]:
        if not self.api_key:
            return {
                "error": "SerpApi API key (SERPAPI_API_KEY) not found or not provided."
            }

        all_params = {**params, "api_key": self.api_key}

        try:
            response = await self._client.get(self.BASE_URL, params=all_params)
            response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
            return response.json()
        except HTTPStatusError as e:
            error_body_details = e.response.text
            try:
                error_body_details = e.response.json()
            except json.JSONDecodeError:
                pass
            return {
                "error": f"SerpApi HTTP error: {e.response.status_code} - {e.response.reason_phrase}",
                "details": error_body_details,
            }
        except RequestError as e:
            return {"error": f"SerpApi request failed: {str(e)}"}
        except json.JSONDecodeError as e:
            return {"error": f"SerpApi failed to decode JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error during SerpApi request: {str(e)}"}

    async def search(self, keyword: str, num_results: int = 10):
        params = {
            "engine": "google_light",
            "q": keyword,
            "num": num_results,
            "gl": self.gl,
            "hl": self.hl,
        }
        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        # Check for SerpApi's own error reporting
        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_news(
        self,
        keyword: str,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        params = {
            "engine": "google_news",
            "q": keyword,
            "num": num_results,
            "gl": self.gl,
            "hl": self.hl,
        }

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_images(
        self,
        keyword: str | None = None,
        image_url: str | None = None,
        num_results: int = 20,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        if not keyword and not image_url:
            return {
                "error": "One of 'keyword' or 'image_url' is required for image search."
            }

        params = {
            "engine": "google_images",
            "num": num_results,
            "gl": self.gl,
            "hl": self.hl,
        }
        if keyword:
            params["q"] = keyword
        if image_url:
            params["engine"] = "google_reverse_image"
            params["image_url"] = image_url
            if "q" in params:
                del params["q"]

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_finance(
        self, query: str, window: str | None = None
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Retrieves financial data using SerpApi's Google Finance engine.
        """
        params = {
            "engine": "google_finance",
            "q": query,
            "gl": self.gl,
            "hl": self.hl,
        }
        if window:
            params["window"] = window

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            # This case handles critical errors from _make_request (e.g., API key missing, network issue)
            return result_data

        # Check for SerpApi's own error reporting within a successful HTTP response
        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        # SerpApi might also return an error message directly at the top level
        serpapi_error_message = isinstance(result_data.get("error"), str)

        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_jobs(
        self, query: str, location: str | None = None, num_results: int = 10
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches for jobs using SerpApi's Google Jobs engine.
        """
        params = {
            "engine": "google_jobs",
            "q": query,
            "hl": self.hl,
            "gl": self.gl,
        }
        if location:
            params["location"] = location

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_lens(
        self, image_url: str, country: str | None = None, num_results: int = 10
    ) -> Union[Dict[str, Any], ErrorResponse]:
        """
        Searches with an image URL using SerpApi's Google Lens engine.
        """
        params = {
            "engine": "google_lens",
            "url": image_url,
            "hl": self.hl,
            # "gl" is supported by google_lens for country, but SerpApi uses "country" for this engine
        }
        if country:
            params["country"] = country  # SerpApi specific parameter for Google Lens

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_maps(
        self,
        query: str,
        ll: str | None = None,
        map_type: str = "search",
        data_id: str | None = None,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches Google Maps using SerpApi.
        If map_type is 'place', data_id is required and query/ll are ignored by SerpApi.
        If map_type is 'search', query is required.
        """
        params = {
            "engine": "google_maps",
            "hl": self.hl,
            "gl": self.gl,  # Standard country code for Google Maps
        }

        if map_type == "place":
            if not data_id:
                return {
                    "error": "data_id is required for map_type 'place' in SerpApiProvider."
                }
            params["type"] = "place"
            params["data_id"] = data_id
            # For 'place' type, 'q' and 'll' are not primary, data_id is key
        elif map_type == "search":
            if not query:
                return {
                    "error": "query is required for map_type 'search' in SerpApiProvider."
                }
            params["type"] = "search"
            params["q"] = query
            if ll:
                params["ll"] = ll  # Format like "@40.7455096,-74.0083012,14z"
        else:
            return {"error": f"Unsupported map_type: {map_type}"}

        result_data = await self._make_request(params)

        if (
            "error" in result_data
            and not isinstance(
                result_data.get("search_metadata"),
                dict,  # serpapi returns error this way
            )
            and not result_data.get("place_results")
        ):  # For place type, error might not have search_metadata
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)

        # Place results have a different structure and error reporting sometimes
        if (
            map_type == "place"
            and not result_data.get("place_results")
            and not serpapi_error_message
        ):
            # If it's a place search and no place_results, and no explicit error, it's likely an issue.
            if (
                not serpapi_error_status and not serpapi_error_message
            ):  # If no explicit error, craft one
                raise ValueError(
                    "SerpApi returned no place_results for the given data_id."
                )

        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def search_shopping(
        self,
        query: str,
        country: str | None = None,
        domain: str | None = None,
        min_price: int | None = None,
        max_price: int | None = None,
        condition: str | None = None,
        sort_by: str | None = None,
        num_results: int = 10,
    ) -> Union[List[Dict[str, Any]], ErrorResponse]:
        """
        Searches Google Shopping using SerpApi.
        Ref: https://serpapi.com/google-shopping-api
        Ref for tbs: https://serpapi.com/google-tbs-api (though shopping has specific tbs patterns)
        """
        params = {
            "engine": "google_shopping",
            "q": query,
            "hl": self.hl,
            # num_results is handled by slicing the response later
        }
        if country:
            params["gl"] = country
        if domain:
            params["google_domain"] = domain

        tbs_parts = []
        # For price and condition, tbs often starts with "mr:1,price:1" or similar
        # However, SerpApi examples for Shopping use more direct tbs components for these.

        if min_price is not None or max_price is not None or condition or sort_by:
            # Base for many shopping filters, though individual components are often enough
            # tbs_parts.append("mr:1") # Let's try without mr:1 first, some filters apply directly
            pass  # placeholder

        if min_price is not None:
            tbs_parts.append(f"ppr_min:{min_price}")
            if "price:1" not in tbs_parts:
                tbs_parts.insert(0, "price:1")  # Ensure price:1 is included for ppr_*
            if "mr:1" not in tbs_parts:
                tbs_parts.insert(0, "mr:1")

        if max_price is not None:
            tbs_parts.append(f"ppr_max:{max_price}")
            if "price:1" not in tbs_parts:
                tbs_parts.insert(0, "price:1")
            if "mr:1" not in tbs_parts:
                tbs_parts.insert(0, "mr:1")

        if condition:
            # SerpApi docs also mention tbs=condition:new directly
            # Let's use the direct approach first if it aligns with their tbs builder. Example: &tbs=condition:new
            # For now, I will use condition directly, as SerpApi might handle it.
            # Based on https://serpapi.com/google-shopping-filters, &tbs=p_cond:new or &tbs=p_cond:used
            # Let's go with p_cond for now as it seems more specific for shopping products.
            if condition.lower() == "new":
                tbs_parts.append("p_cond:new")
            elif condition.lower() == "used":
                tbs_parts.append("p_cond:used")
            elif (
                condition.lower() == "refurbished"
            ):  # SerpApi might not have a direct p_cond for refurbished, often grouped with used
                tbs_parts.append(
                    "p_cond:used"
                )  # Or could try general condition:refurbished if supported

        if (
            sort_by
        ):  # Common values: "r" (relevance), "p" (price asc), "pd" (price desc)
            tbs_parts.append(f"sort:{sort_by}")

        if tbs_parts:
            params["tbs"] = ",".join(tbs_parts)

        result_data = await self._make_request(params)

        if "error" in result_data and not isinstance(
            result_data.get("search_metadata"), dict
        ):
            return result_data

        serpapi_error_status = (
            result_data.get("search_metadata", {}).get("status") == "Error"
        )
        serpapi_error_message = isinstance(result_data.get("error"), str)
        if serpapi_error_status or serpapi_error_message:
            raise ValueError(
                result_data.get("error", f"SerpApi returned an error: {result_data}")
            )

        return _remove_base64_images(result_data)

    async def close(self) -> None:
        await self._client.aclose()
