"""Metadata resolution across the Hub's redirect chain.

``hf_head_metadata`` sent one HEAD with ``follow_redirects=False`` and required
the ETag on that first response. For most repos the first hop is the
resolve-cache redirect and carries ``x-linked-etag``. For a repo that has been
RENAMED the first hop is the rename redirect, which carries no ETag and no
commit — both appear one hop later — so the whole download aborted on the first
file with "No ETag received from Hugging Face".

Observed on `cross-encoder/ms-marco-MiniLM-L-6-v2` (renamed to `...-L6-v2`),
and on two repos that are a shipped node's own default model:
`runwayml/stable-diffusion-v1-5` and
`bosonai/higgs-audio-v2-generation-3B-base`.

The chains below are the real ones, captured with `curl -sIL`.
"""

import httpx
import pytest

from nodetool.integrations.huggingface import async_downloader as dl
from nodetool.integrations.huggingface.async_downloader import hf_head_metadata

OLD_NAME = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/.gitattributes"
NEW_NAME = "/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/main/.gitattributes"
RESOLVE_CACHE = (
    "/api/resolve-cache/models/cross-encoder/ms-marco-MiniLM-L6-v2/"
    "233902d25c440f23af6f7d6e94d2946bac0bee0a/.gitattributes"
)
COMMIT = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
ETAG = "cf6d51fc9b1a671c35e92d6bd009880937aaa12d"
CDN = "https://us.aws.cdn.hf.co/xet-bridge-us/deadbeef/cafe?Expires=1&Signature=x"


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_renamed_repo_resolves_on_the_second_hop():
    """The rename hop has no ETag; the metadata comes from the hop that does."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path.endswith("ms-marco-MiniLM-L-6-v2/resolve/main/.gitattributes"):
            # Hop 1: rename redirect. No etag, no commit, no size.
            return httpx.Response(307, headers={"Location": NEW_NAME})
        # Hop 2: resolve-cache redirect, carrying the metadata.
        return httpx.Response(
            307,
            headers={
                "Location": RESOLVE_CACHE,
                "X-Linked-Etag": f'"{ETAG}"',
                "X-Linked-Size": "790",
                "X-Repo-Commit": COMMIT,
                "Accept-Ranges": "bytes",
            },
        )

    async with _client(handler) as client:
        meta = await hf_head_metadata(client, OLD_NAME)

    assert len(seen) == 2, seen
    assert meta.etag == ETAG
    assert meta.commit_hash == COMMIT
    assert meta.size == 790
    assert meta.accept_ranges is True
    assert meta.url == "https://huggingface.co" + RESOLVE_CACHE


@pytest.mark.asyncio
async def test_commit_comes_from_the_hop_that_supplied_the_etag():
    """A commit from a different hop would cache files under the wrong snapshot.

    Hop 1 here carries a stale commit and no ETag. Pairing that commit with the
    ETag resolved later is worse than today's hard failure — the files would
    land in a snapshot directory naming a commit they did not come from.
    """
    stale = "0000000000000000000000000000000000000000"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("ms-marco-MiniLM-L-6-v2/resolve/main/.gitattributes"):
            return httpx.Response(
                307, headers={"Location": NEW_NAME, "X-Repo-Commit": stale}
            )
        return httpx.Response(
            307,
            headers={
                "Location": RESOLVE_CACHE,
                "X-Linked-Etag": f'"{ETAG}"',
                "X-Repo-Commit": COMMIT,
            },
        )

    async with _client(handler) as client:
        meta = await hf_head_metadata(client, OLD_NAME)

    assert meta.commit_hash == COMMIT
    assert meta.commit_hash != stale


@pytest.mark.asyncio
async def test_single_hop_repo_still_takes_one_request():
    """The common case must not change: one HEAD, metadata off that response."""
    seen: list[str] = []
    cache = (
        "/api/resolve-cache/models/google/vit-base-patch16-224/"
        "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3/config.json"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(
            307,
            headers={
                "Location": cache,
                "X-Linked-Etag": '"21c61d3f8f3e50137a8c5fdd5bc9b085e286315a"',
                "X-Linked-Size": "69665",
                "X-Repo-Commit": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
                "Accept-Ranges": "bytes",
            },
        )

    async with _client(handler) as client:
        meta = await hf_head_metadata(
            client, "https://huggingface.co/google/vit-base-patch16-224/resolve/main/config.json"
        )

    assert len(seen) == 1, seen
    assert meta.etag == "21c61d3f8f3e50137a8c5fdd5bc9b085e286315a"
    assert meta.size == 69665


@pytest.mark.asyncio
async def test_cdn_handoff_is_not_followed():
    """The hop that leaves huggingface.co is the CDN hand-off. Stop before it.

    Its own ETag is a different hash than the one the cache layout is keyed on,
    and the URL is signed. This is what `follow_redirects=False` was protecting
    and the fix must keep protecting.
    """
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "huggingface.co":
            return httpx.Response(
                302,
                headers={
                    "Location": CDN,
                    "X-Linked-Etag": '"1cea07110a4a47edc51420b2dda6f3b8b58e7256e"',
                    "X-Linked-Size": "346293852",
                    "X-Repo-Commit": "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3",
                },
            )
        raise AssertionError(f"followed the CDN hand-off to {request.url.host}")

    async with _client(handler) as client:
        meta = await hf_head_metadata(
            client,
            "https://huggingface.co/google/vit-base-patch16-224/resolve/main/model.safetensors",
        )

    assert seen == ["huggingface.co"]
    assert meta.etag == "1cea07110a4a47edc51420b2dda6f3b8b58e7256e"
    assert meta.url == CDN


@pytest.mark.asyncio
async def test_cross_origin_hop_without_etag_is_not_followed():
    """A redirect off huggingface.co is refused even when we still have no ETag.

    Following it would send the Authorization header to another host.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host != "huggingface.co":
            raise AssertionError(f"left huggingface.co for {request.url.host}")
        return httpx.Response(307, headers={"Location": "https://evil.invalid/steal"})

    async with _client(handler) as client:
        with pytest.raises(RuntimeError, match="No ETag"):
            await hf_head_metadata(client, OLD_NAME, token="hf_secret")


@pytest.mark.asyncio
async def test_redirect_loop_is_bounded():
    """A chain that never supplies an ETag stops instead of hanging."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(307, headers={"Location": f"/hop/{len(seen)}"})

    async with _client(handler) as client:
        with pytest.raises(RuntimeError, match="Too many redirects"):
            await hf_head_metadata(client, OLD_NAME)

    assert len(seen) == dl._MAX_METADATA_REDIRECTS + 1


@pytest.mark.asyncio
async def test_token_is_sent_on_every_same_origin_hop():
    """The rename hop is authenticated too — a gated repo 401s otherwise."""
    auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        auth.append(request.headers.get("Authorization"))
        if len(auth) == 1:
            return httpx.Response(307, headers={"Location": NEW_NAME})
        return httpx.Response(
            307, headers={"Location": RESOLVE_CACHE, "X-Linked-Etag": f'"{ETAG}"'}
        )

    async with _client(handler) as client:
        await hf_head_metadata(client, OLD_NAME, token="hf_secret")

    assert auth == ["Bearer hf_secret", "Bearer hf_secret"]


@pytest.mark.asyncio
async def test_unauthorized_on_a_later_hop_still_raises_permission_error():
    """The 401/403 mapping must survive the loop, not just the first hop."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("ms-marco-MiniLM-L-6-v2/resolve/main/.gitattributes"):
            return httpx.Response(307, headers={"Location": NEW_NAME})
        return httpx.Response(401)

    async with _client(handler) as client:
        with pytest.raises(PermissionError) as exc:
            await hf_head_metadata(client, OLD_NAME, token="hf_secret")

    # The message reports presence, never the value.
    assert "hf_secret" not in str(exc.value)
    assert "Token present: True" in str(exc.value)
