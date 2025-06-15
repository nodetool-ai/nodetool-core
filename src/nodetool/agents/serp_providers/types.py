from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

from nodetool.metadata.types import BaseType


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


class OrganicResult(BaseType):
    """Path: organic_results[] (top-level array)"""

    type: Literal["organic_result"] = "organic_result"
    position: int
    title: str
    link: str
    redirect_link: Optional[str] = None
    displayed_link: str
    date: Optional[str] = None
    snippet: str
    snippet_highlighted_words: Optional[List[str]] = None
    # sitelinks: Optional["InlineSiteLinks"] = None
    # rich_snippet: Optional["RichSnippet"] = None
    # about_this_result: Optional["AboutThisResult"] = None
    # about_page_link: Optional[str] = None
    # about_page_serpapi_link: Optional[str] = None
    # cached_page_link: Optional[str] = None
    # related_pages_link: Optional[str] = None
    thumbnail: Optional[str] = None
    # displayed_results: Optional[str] = None


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


class NewsResult(BaseType):
    """Path: news_results[] (main news results array)"""

    type: Literal["news_result"] = "news_result"

    position: int
    title: str | None = None
    link: str
    thumbnail: str | None = None
    date: str


class ImageResult(BaseType):
    """Path: images_results[] (main images results array)"""

    type: Literal["image_result"] = "image_result"

    position: int
    thumbnail: str
    original: str
    original_width: int
    original_height: int
    is_product: bool
    source: str
    title: str
    link: str


class JobResult(BaseType):
    """Path: jobs_results[] (main jobs results array)"""

    type: Literal["job_result"] = "job_result"

    title: str | None = None
    company_name: str | None = None
    location: str | None = None
    via: str | None = None
    share_link: str | None = None
    thumbnail: str | None = None
    extensions: list[str] | None = None


class ViualMatchResult(BaseType):
    """Path: visual_matches[] (main visual matches results array)"""

    type: Literal["visual_match_result"] = "visual_match_result"
    position: int
    title: str | None = None
    link: str | None = None
    thumbnail: str | None = None
    thumbnail_width: int | None = None
    thumbnail_height: int | None = None
    image: str | None = None
    image_width: int | None = None
    image_height: int | None = None


class LocalResult(BaseType):
    """Path: local_results[] (main local results array)"""

    type: Literal["local_result"] = "local_result"
    position: int
    title: str | None = None
    place_id: str | None = None
    data_id: str | None = None
    data_cid: str | None = None
    reviews_link: str | None = None
    photos_link: str | None = None
    gps_coordinates: Dict[str, float] | None = None
    place_id_search: str | None = None
    provider_id: str | None = None
    rating: float | None = None
    reviews: int | None = None
    price: str | None = None
    types: list[str] | None = None
    address: str | None = None
    open_state: str | None = None
    hours: str | None = None
    operating_hours: Dict[str, str] | None = None
    phone: str | None = None
    website: str | None = None
    description: str | None = None
    thumbnail: str | None = None


class ShoppingResult(BaseType):
    """Path: shopping_results[] (main shopping results array)"""

    type: Literal["shopping_result"] = "shopping_result"
    position: int
    title: str | None = None
    link: str | None = None
    product_link: str | None = None
    product_id: str | None = None
    source: str | None = None
    source_icon: str | None = None
    extensions: list[str] | None = None
    badge: str | None = None
    thumbnail: str | None = None
    tag: str | None = None
    delivery: str | None = None
    price: str | None = None
    extracted_price: float | None = None
    old_price: str | None = None
    extracted_old_price: float | None = None
    rating: float | None = None
    reviews: int | None = None
    store_rating: float | None = None
    store_reviews: int | None = None


class GoogleShoppingResponse(BaseModel):
    shopping_results: list[ShoppingResult] = Field(default_factory=list)


class GoogleNewsResponse(BaseModel):
    news_results: list[NewsResult] = Field(default_factory=list)


class GoogleImagesResponse(BaseModel):
    images_results: list[ImageResult] = Field(default_factory=list)


class GoogleJobsResponse(BaseModel):
    jobs_results: list[JobResult] = Field(default_factory=list)


class GoogleLensResponse(BaseModel):
    visual_matches: list[ViualMatchResult] = Field(default_factory=list)


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
