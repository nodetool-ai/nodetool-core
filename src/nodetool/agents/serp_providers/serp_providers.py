import abc
from typing import Any

ErrorResponse = dict[str, Any]


class SerpProvider(abc.ABC):
    """
    Abstract base class for a SERP (Search Engine Results Page) provider.
    Defines a common interface for fetching search results from various backends.
    """

    @abc.abstractmethod
    async def search(self, keyword: str, num_results: int = 10) -> Any:
        """
        Perform an organic web search.

        Args:
            keyword: The search query.
            num_results: The desired number of results. Defaults to 10.
        """
        pass

    @abc.abstractmethod
    async def search_news(self, keyword: str, num_results: int = 10) -> Any:
        """
        Perform a news search.

        Args:
            keyword: The search query.
            num_results: The desired number of results. Defaults to 10.

        """
        pass

    @abc.abstractmethod
    async def search_images(
        self,
        keyword: str | None = None,
        image_url: str | None = None,
        num_results: int = 10,
    ) -> Any:
        """
        Perform an image search.
        One of 'keyword' or 'image_url' must be provided.

        Args:
            keyword: The search query for images.
            image_url: URL of an image to use for reverse image search.
            num_results: The desired number of image results. Defaults to 10.

        """
        pass

    @abc.abstractmethod
    async def search_finance(self, query: str, window: str | None = None) -> Any:
        """
        Retrieves financial data.
        """
        pass

    @abc.abstractmethod
    async def search_jobs(self, query: str, location: str | None = None, num_results: int = 10) -> Any:
        """
        Perform a job search.
        """
        pass

    @abc.abstractmethod
    async def search_lens(
        self,
        image_url: str,
        country: str | None = None,
        num_results: int = 10,
    ) -> Any:
        """
        Perform a lens search.
        """
        pass

    @abc.abstractmethod
    async def search_maps(
        self,
        query: str,
        ll: str | None = None,
        map_type: str = "search",
        data_id: str | None = None,
        num_results: int = 10,
    ) -> Any:
        """
        Perform a maps search.
        """
        pass

    @abc.abstractmethod
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
    ) -> Any:
        """
        Perform a shopping search.
        """
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """
        Clean up any resources (e.g., close HTTP clients).
        To be called when the provider is no longer needed, for example,
        when exiting an `async with` block.
        """
        pass

    async def __aenter__(self):
        """Allows the provider to be used as an asynchronous context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensures resources are cleaned up when exiting an `async with` block."""
        await self.close()
