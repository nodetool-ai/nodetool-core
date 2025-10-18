from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from nodetool.metadata.types import BaseType


class GoogleNewsResponse(BaseModel):
    """TOP-LEVEL: Root model containing all news results and metadata"""

    top_stories_link: Optional["TopStoriesLink"] = None
    title: Optional[str] = None
    news_results: Optional[List["NewsResult"]] = None
    menu_links: Optional[List["MenuLink"]] = None
    sub_menu_links: Optional[List["SubMenuLink"]] = None
    related_topics: Optional[List["RelatedTopic"]] = None
    related_publications: Optional[List["RelatedPublication"]] = None


# =============================================================================
# LEVEL 1: TOP-LEVEL FIELDS (direct children of GoogleNewsResponse)
# =============================================================================


class TopStoriesLink(BaseModel):
    """Path: top_stories_link (top-level field)"""

    topic_token: Optional[str] = None
    serpapi_link: Optional[str] = None


class NewsResult(BaseType):
    """Path: news_results[] (main news results array)"""

    type: Literal["news_result"] = "news_result"
    position: Optional[int] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional["NewsSource"] = None
    author: Optional["NewsAuthor"] = None
    link: Optional[str] = None
    thumbnail: Optional[str] = None
    thumbnail_small: Optional[str] = None
    type_field: Optional[str] = Field(
        None, alias="type"
    )  # e.g., 'Opinion', 'Local coverage'
    video: Optional[bool] = None
    topic_token: Optional[str] = None
    story_token: Optional[str] = None
    serpapi_link: Optional[str] = None
    date: Optional[str] = None
    related_topics: Optional[List["NewsRelatedTopic"]] = None
    highlight: Optional["NewsHighlight"] = None
    stories: Optional[List["NewsStory"]] = None


class MenuLink(BaseModel):
    """Path: menu_links[] (top-level array)"""

    title: Optional[str] = None
    topic_token: Optional[str] = None
    serpapi_link: Optional[str] = None


class SubMenuLink(BaseModel):
    """Path: sub_menu_links[] (top-level array)"""

    title: Optional[str] = None
    section_token: Optional[str] = None
    topic_token: Optional[str] = None
    serpapi_link: Optional[str] = None


class RelatedTopic(BaseModel):
    """Path: related_topics[] (top-level array)"""

    title: Optional[str] = None
    topic_token: Optional[str] = None
    serpapi_link: Optional[str] = None
    thumbnail: Optional[str] = None


class RelatedPublication(BaseModel):
    """Path: related_publications[] (top-level array)"""

    title: Optional[str] = None
    publication_token: Optional[str] = None
    serpapi_link: Optional[str] = None
    thumbnail: Optional[str] = None


# =============================================================================
# LEVEL 2: NESTED FIELDS WITHIN NEWS RESULTS
# =============================================================================


class NewsSource(BaseModel):
    """Path: news_results[].source"""

    title: Optional[str] = None
    name: Optional[str] = None
    icon: Optional[str] = None
    authors: Optional[List[str]] = None


class NewsAuthor(BaseModel):
    """Path: news_results[].author"""

    thumbnail: Optional[str] = None
    name: Optional[str] = None
    handle: Optional[str] = None  # X (Twitter) username


class NewsRelatedTopic(BaseModel):
    """Path: news_results[].related_topics[]"""

    position: Optional[int] = None
    name: Optional[str] = None
    topic_token: Optional[str] = None
    serpapi_link: Optional[str] = None


class NewsHighlight(BaseModel):
    """Path: news_results[].highlight (similar to NewsResult but without some fields)"""

    position: Optional[int] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional["NewsSource"] = None
    author: Optional["NewsAuthor"] = None
    link: Optional[str] = None
    thumbnail: Optional[str] = None
    thumbnail_small: Optional[str] = None
    type_field: Optional[str] = Field(None, alias="type")
    video: Optional[bool] = None
    topic_token: Optional[str] = None
    story_token: Optional[str] = None
    serpapi_link: Optional[str] = None
    date: Optional[str] = None


class NewsStory(BaseModel):
    """Path: news_results[].stories[] (similar to NewsResult but without some fields)"""

    position: Optional[int] = None
    title: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional["NewsSource"] = None
    author: Optional["NewsAuthor"] = None
    link: Optional[str] = None
    thumbnail: Optional[str] = None
    thumbnail_small: Optional[str] = None
    type_field: Optional[str] = Field(None, alias="type")
    video: Optional[bool] = None
    topic_token: Optional[str] = None
    story_token: Optional[str] = None
    serpapi_link: Optional[str] = None
    date: Optional[str] = None
