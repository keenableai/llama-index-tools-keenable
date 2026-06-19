"""Keenable tool spec for LlamaIndex: web search and page fetch."""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.schema import Document
from llama_index.core.tools.tool_spec.base import BaseToolSpec

from llama_index.tools.keenable._client import (
    KeenableError,
    keenable_get,
    keenable_post,
    reject_private_fetch_target,
    resolve_api_key,
)


class KeenableToolSpec(BaseToolSpec):
    """Keenable web-search and page-fetch tools for LlamaIndex agents.

    Keyless by default: with no API key the keyless public endpoints are used.
    Provide an API key (constructor arg or the ``KEENABLE_API_KEY`` environment
    variable) to use the authenticated endpoints, required for ``mode="realtime"``
    and for higher rate limits.

    The API endpoint is read from ``KEENABLE_API_URL`` (HTTPS enforced), never a
    function argument, so the model cannot redirect requests.
    """

    spec_functions = ["search", "fetch"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = "pro",
        timeout: float = 30.0,
    ) -> None:
        """Initialize the tool spec.

        Args:
            api_key: Optional Keenable API key. Falls back to ``KEENABLE_API_KEY``;
                with no key the keyless public endpoints are used.
            mode: Default search mode, ``"pro"`` (deeper) or ``"realtime"`` (low
                latency). ``"realtime"`` requires an API key.
            timeout: Per-request timeout in seconds.

        """
        self._api_key = resolve_api_key(api_key)
        self._mode = mode
        self._timeout = timeout

    def search(
        self,
        query: str,
        site: Optional[str] = None,
        published_after: Optional[str] = None,
        published_before: Optional[str] = None,
        acquired_after: Optional[str] = None,
        acquired_before: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> List[Document]:
        """
        Search the web with Keenable, a search engine built for AI agents.

        Args:
            query: The search query to run.
            site: Restrict results to a single domain, e.g. ``"github.com"``.
            published_after: Only pages published on/after this date (YYYY-MM-DD).
            published_before: Only pages published on/before this date (YYYY-MM-DD).
            acquired_after: Only pages indexed on/after this date (YYYY-MM-DD).
            acquired_before: Only pages indexed on/before this date (YYYY-MM-DD).
            mode: Override the default search mode for this query (``"pro"`` or
                ``"realtime"``).

        Returns:
            A list of Documents, one per result. The text is the result snippet;
            ``metadata`` carries ``url``, ``title``, ``description``,
            ``published_at`` and ``acquired_at``.

        """
        payload: dict = {"query": query, "mode": mode or self._mode}
        for field, value in (
            ("site", site),
            ("published_after", published_after),
            ("published_before", published_before),
            ("acquired_after", acquired_after),
            ("acquired_before", acquired_before),
        ):
            if value:
                payload[field] = value

        data = keenable_post(
            "/v1/search/public", "/v1/search", payload, self._api_key, self._timeout
        )
        results = data.get("results")
        if not isinstance(results, list):
            msg = f"Unexpected response from the Keenable search API: {data!r}"
            raise KeenableError(msg)

        documents = []
        for result in results:
            text = result.get("description") or result.get("title") or ""
            documents.append(Document(text=text, metadata=dict(result)))
        return documents

    def fetch(self, url: str) -> List[Document]:
        """
        Fetch a web page via Keenable and return its main content as markdown.

        Use this to read a page found via :meth:`search`. Rejects non-``http(s)``
        and private/internal URLs before sending.

        Args:
            url: The URL of the page to fetch.

        Returns:
            A single-element list with a Document whose text is the page content;
            ``metadata`` carries ``url``, ``title`` and any other metadata the
            page exposes (``description``, ``author``, ``published_at``).

        """
        if not url.lower().startswith(("http://", "https://")):
            msg = f"Refusing to fetch a non-http(s) URL: {url!r}"
            raise KeenableError(msg)
        reject_private_fetch_target(url)

        data = keenable_get(
            "/v1/fetch/public", "/v1/fetch", {"url": url}, self._api_key, self._timeout
        )
        text = data.get("content") or ""
        return [Document(text=text, metadata=dict(data))]
