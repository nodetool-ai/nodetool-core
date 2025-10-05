from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

from nodetool.metadata.types import (
    BaseType,
    OrganicResult,
    NewsResult,
    ImageResult,
    JobResult,
    VisualMatchResult,
    LocalResult,
    ShoppingResult,
)

__all__ = [
    "OrganicResult",
    "NewsResult",
    "ImageResult",
    "JobResult",
    "VisualMatchResult",
    "LocalResult",
    "ShoppingResult",
    "GoogleSearchResponse",
    "GoogleNewsResponse",
    "GoogleImagesResponse",
    "GoogleJobsResponse",
    "GoogleLensResponse",
    "GoogleMapsResponse",
    "GoogleShoppingResponse",
]


class GoogleSearchResponse(BaseModel):
    """TOP-LEVEL: Root model containing all search results and metadata"""

    search_metadata: "SearchMetadata"
    search_parameters: "SearchParameters"
    search_information: "SearchInformation"
    organic_results: Optional[List["OrganicResult"]] = None


# =============================================================================
# LEVEL 1: TOP-LEVEL FIELDS (direct children of GoogleSearchResponse)
# =============================================================================


class SearchMetadata(BaseModel):
    """Path: search_metadata (top-level field)"""

    id: str
    status: str
    json_endpoint: str
    created_at: str
    processed_at: str
    google_url: Optional[str] = None
    raw_html_file: str
    total_time_taken: float


class SearchParameters(BaseModel):
    """Path: search_parameters (top-level field)"""

    engine: str
    q: str
    location_requested: Optional[str] = None
    location_used: Optional[str] = None
    google_domain: str
    hl: str
    gl: str
    device: str


class SearchInformation(BaseModel):
    """Path: search_information (top-level field)"""

    organic_results_state: str
    query_displayed: str
    total_results: Optional[int] = None
    time_taken_displayed: Optional[float] = None


class Pagination(BaseModel):
    """Path: pagination (top-level field)"""

    current: int
    next: str
    other_pages: Dict[str, str]


class SerpapiPagination(BaseModel):
    """Path: serpapi_pagination (top-level field)"""

    current: int
    next_link: Optional[str] = None
    next: str
    other_pages: Optional[Dict[str, str]] = None


class GoogleShoppingResponse(BaseModel):
    shopping_results: list[ShoppingResult] = Field(default_factory=list)


class GoogleNewsResponse(BaseModel):
    news_results: list[NewsResult] = Field(default_factory=list)


class GoogleImagesResponse(BaseModel):
    images_results: list[ImageResult] = Field(default_factory=list)


class GoogleJobsResponse(BaseModel):
    jobs_results: list[JobResult] = Field(default_factory=list)


class GoogleLensResponse(BaseModel):
    visual_matches: list[VisualMatchResult] = Field(default_factory=list)


class GoogleMapsResponse(BaseModel):
    local_results: list[LocalResult] = Field(default_factory=list)


if __name__ == "__main__":
    # You can test the schema with your JSON data
    import json

    search_results_file = Path(__file__).parent / "search_results.json"
    with open(search_results_file, "r") as f:
        data = json.load(f)
    try:
        search_results = GoogleSearchResponse(**data)
        print("Validation successful!")
        print(search_results)
    except Exception as e:
        print(f"Validation error: {e}")
