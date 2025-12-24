from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from nodetool.metadata.types import BaseType


class GoogleNewsResponse(BaseModel):
    """TOP-LEVEL: Root model containing all news results and metadata"""

    top_stories_link: TopStoriesLink | None = None
    title: str | None = None
    news_results: list[NewsResult] | None = None
    menu_links: list[MenuLink] | None = None
    sub_menu_links: list[SubMenuLink] | None = None
    related_topics: list[RelatedTopic] | None = None
    related_publications: list[RelatedPublication] | None = None


# =============================================================================
# LEVEL 1: TOP-LEVEL FIELDS (direct children of GoogleNewsResponse)
# =============================================================================


class TopStoriesLink(BaseModel):
    """Path: top_stories_link (top-level field)"""

    topic_token: str | None = None
    serpapi_link: str | None = None


class NewsResult(BaseType):
    """Path: news_results[] (main news results array)"""

    type: Literal["news_result"] = "news_result"
    position: int | None = None
    title: str | None = None
    snippet: str | None = None
    source: NewsSource | None = None
    author: NewsAuthor | None = None
    link: str | None = None
    thumbnail: str | None = None
    thumbnail_small: str | None = None
    type_field: str | None = Field(None, alias="type")  # e.g., 'Opinion', 'Local coverage'
    video: bool | None = None
    topic_token: str | None = None
    story_token: str | None = None
    serpapi_link: str | None = None
    date: str | None = None
    related_topics: list[NewsRelatedTopic] | None = None
    highlight: NewsHighlight | None = None
    stories: list[NewsStory] | None = None


class MenuLink(BaseModel):
    """Path: menu_links[] (top-level array)"""

    title: str | None = None
    topic_token: str | None = None
    serpapi_link: str | None = None


class SubMenuLink(BaseModel):
    """Path: sub_menu_links[] (top-level array)"""

    title: str | None = None
    section_token: str | None = None
    topic_token: str | None = None
    serpapi_link: str | None = None


class RelatedTopic(BaseModel):
    """Path: related_topics[] (top-level array)"""

    title: str | None = None
    topic_token: str | None = None
    serpapi_link: str | None = None
    thumbnail: str | None = None


class RelatedPublication(BaseModel):
    """Path: related_publications[] (top-level array)"""

    title: str | None = None
    publication_token: str | None = None
    serpapi_link: str | None = None
    thumbnail: str | None = None


# =============================================================================
# LEVEL 2: NESTED FIELDS WITHIN NEWS RESULTS
# =============================================================================


class NewsSource(BaseModel):
    """Path: news_results[].source"""

    title: str | None = None
    name: str | None = None
    icon: str | None = None
    authors: list[str] | None = None


class NewsAuthor(BaseModel):
    """Path: news_results[].author"""

    thumbnail: str | None = None
    name: str | None = None
    handle: str | None = None  # X (Twitter) username


class NewsRelatedTopic(BaseModel):
    """Path: news_results[].related_topics[]"""

    position: int | None = None
    name: str | None = None
    topic_token: str | None = None
    serpapi_link: str | None = None


class NewsHighlight(BaseModel):
    """Path: news_results[].highlight (similar to NewsResult but without some fields)"""

    position: int | None = None
    title: str | None = None
    snippet: str | None = None
    source: NewsSource | None = None
    author: NewsAuthor | None = None
    link: str | None = None
    thumbnail: str | None = None
    thumbnail_small: str | None = None
    type_field: str | None = Field(None, alias="type")
    video: bool | None = None
    topic_token: str | None = None
    story_token: str | None = None
    serpapi_link: str | None = None
    date: str | None = None


class NewsStory(BaseModel):
    """Path: news_results[].stories[] (similar to NewsResult but without some fields)"""

    position: int | None = None
    title: str | None = None
    snippet: str | None = None
    source: NewsSource | None = None
    author: NewsAuthor | None = None
    link: str | None = None
    thumbnail: str | None = None
    thumbnail_small: str | None = None
    type_field: str | None = Field(None, alias="type")
    video: bool | None = None
    topic_token: str | None = None
    story_token: str | None = None
    serpapi_link: str | None = None
    date: str | None = None
