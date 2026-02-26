from __future__ import annotations

import os
import sys
from typing import Any

from rich.console import Console
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_package_version

# Create console instance
console = Console(stderr=True)

def run_serve(
    host: str,
    port: int,
    static_folder: str | None = None,
    reload: bool = False,
    force_fp16: bool = False,
    mode: str | None = None,
    auth_provider: str | None = None,
    enable_default_api: bool | None = None,
    enable_openai: bool | None = None,
    enable_deploy_admin: bool | None = None,
    enable_deploy_collections: bool | None = None,
    enable_deploy_storage: bool | None = None,
    enable_main_ws: bool | None = None,
    enable_terminal_ws: bool | None = None,
    enable_hf_download_ws: bool | None = None,
    mcp: bool = False,
    apps_folder: str | None = None,
    production: bool = False,
    ui_url: str | None = None,
    ui: bool = False,
    verbose: bool = False,
    mock: bool = False,
):
    """Run the FastAPI backend server for the NodeTool platform.

    Serves the REST API, WebSocket endpoints, and optionally static assets or app bundles.

    Use --production to run the production server with full admin routers.
    Use --mock to start with pre-filled test data for development and testing.
    """
    if auth_provider:
        os.environ["AUTH_PROVIDER"] = auth_provider.lower()

    effective_mode = mode.lower() if mode else ("private" if production else "desktop")

    if production:
        _run_production_server(
            host=host,
            port=port,
            reload=reload,
            mode=effective_mode,
            auth_provider=auth_provider,
            enable_default_api=enable_default_api,
            enable_openai=enable_openai,
            enable_deploy_admin=enable_deploy_admin,
            enable_deploy_collections=enable_deploy_collections,
            enable_deploy_storage=enable_deploy_storage,
            enable_main_ws=enable_main_ws,
            enable_terminal_ws=enable_terminal_ws,
            enable_hf_download_ws=enable_hf_download_ws,
            mcp=mcp,
            static_folder=static_folder,
            apps_folder=apps_folder,
            mock=mock,
        )
        return

    _run_development_server(
        host=host,
        port=port,
        reload=reload,
        effective_mode=effective_mode,
        auth_provider=auth_provider,
        enable_default_api=enable_default_api,
        enable_openai=enable_openai,
        enable_deploy_admin=enable_deploy_admin,
        enable_deploy_collections=enable_deploy_collections,
        enable_deploy_storage=enable_deploy_storage,
        enable_main_ws=enable_main_ws,
        enable_terminal_ws=enable_terminal_ws,
        enable_hf_download_ws=enable_hf_download_ws,
        mcp=mcp,
        static_folder=static_folder,
        apps_folder=apps_folder,
        force_fp16=force_fp16,
        ui_url=ui_url,
        ui=ui,
        verbose=verbose,
        mock=mock,
    )


def _run_production_server(
    host: str,
    port: int,
    reload: bool,
    mode: str,
    auth_provider: str | None,
    enable_default_api: bool | None,
    enable_openai: bool | None,
    enable_deploy_admin: bool | None,
    enable_deploy_collections: bool | None,
    enable_deploy_storage: bool | None,
    enable_main_ws: bool | None,
    enable_terminal_ws: bool | None,
    enable_hf_download_ws: bool | None,
    mcp: bool,
    static_folder: str | None,
    apps_folder: str | None,
    mock: bool,
):
    from nodetool.api.run_server import run_server

    if static_folder:
        console.print("[yellow]Warning: --static-folder ignored in production mode[/]")
    if apps_folder:
        console.print("[yellow]Warning: --apps-folder ignored in production mode[/]")
    if mock:
        console.print("[yellow]Warning: --mock ignored in production mode[/]")

    run_kwargs: dict[str, Any] = {
        "host": host,
        "port": port,
        "reload": reload,
    }
    if mode:
        run_kwargs["mode"] = mode
    else:
        os.environ.setdefault("NODETOOL_SERVER_MODE", "private")
    if auth_provider:
        run_kwargs["auth_provider"] = auth_provider
    if enable_default_api is not None:
        run_kwargs["include_default_api_routers"] = enable_default_api
    if enable_openai is not None:
        run_kwargs["include_openai_router"] = enable_openai
    if enable_deploy_admin is not None:
        run_kwargs["include_deploy_admin_router"] = enable_deploy_admin
    if enable_deploy_collections is not None:
        run_kwargs["include_deploy_collection_router"] = enable_deploy_collections
    if enable_deploy_storage is not None:
        run_kwargs["include_deploy_storage_router"] = enable_deploy_storage
    if enable_main_ws is not None:
        run_kwargs["enable_main_ws"] = enable_main_ws
    if enable_terminal_ws is not None:
        run_kwargs["enable_terminal_ws"] = enable_terminal_ws
    if enable_hf_download_ws is not None:
        run_kwargs["enable_hf_download_ws"] = enable_hf_download_ws
    if mcp:
        run_kwargs["enable_mcp"] = mcp
    run_server(**run_kwargs)


def _run_development_server(
    host: str,
    port: int,
    reload: bool,
    effective_mode: str,
    auth_provider: str | None,
    enable_default_api: bool | None,
    enable_openai: bool | None,
    enable_deploy_admin: bool | None,
    enable_deploy_collections: bool | None,
    enable_deploy_storage: bool | None,
    enable_main_ws: bool | None,
    enable_terminal_ws: bool | None,
    enable_hf_download_ws: bool | None,
    mcp: bool,
    static_folder: str | None,
    apps_folder: str | None,
    force_fp16: bool,
    ui_url: str | None,
    ui: bool,
    verbose: bool,
    mock: bool,
):
    from nodetool.api.server import create_app, run_uvicorn_server

    # Configure logging level based on verbose flag
    if verbose:
        from nodetool.config.logging_config import configure_logging

        configure_logging(level="DEBUG")
        os.environ["LOG_LEVEL"] = "DEBUG"
        console.print("[cyan]🐛 Verbose logging enabled (DEBUG level)[/]")

    # Configure mock mode
    if mock:
        console.print("[yellow]🎭 Mock mode enabled - will populate database with test data[/]")
        os.environ["NODETOOL_MOCK_MODE"] = "1"

    try:
        import comfy.cli_args  # type: ignore

        comfy.cli_args.args.force_fp16 = force_fp16
    except ImportError:
        pass

    static_folder = _resolve_static_folder(static_folder, ui_url, ui)

    if not reload:
        app = create_app(
            static_folder=static_folder,
            apps_folder=apps_folder,
            mode=effective_mode,
            auth_provider=auth_provider,
            include_default_api_routers=enable_default_api,
            include_openai_router=enable_openai,
            include_deploy_admin=enable_deploy_admin,
            include_deploy_collection_router=enable_deploy_collections,
            include_deploy_storage=enable_deploy_storage,
            enable_main_ws=enable_main_ws,
            enable_terminal_ws=enable_terminal_ws,
            enable_hf_download_ws=enable_hf_download_ws,
            enable_mcp=mcp,
        )
        if mcp:
            console.print("[green]MCP server enabled at /mcp (endpoints: /mcp/sse, /mcp/messages)[/]")
    else:
        if static_folder:
            raise Exception("static folder and reload are exclusive options")
        if apps_folder:
            raise Exception("apps folder and reload are exclusive options")
        # Pass MCP flag via environment variable for reload mode
        if mcp:
            os.environ["NODETOOL_ENABLE_MCP"] = "1"
            console.print("[green]MCP server enabled at /mcp (endpoints: /mcp/sse, /mcp/messages)[/]")
        app = "nodetool.api.app:app"

    run_uvicorn_server(app=app, host=host, port=port, reload=reload)


def _resolve_static_folder(static_folder: str | None, ui_url: str | None, ui: bool) -> str | None:
    # Handle UI download if requested
    if ui or ui_url:
        if ui and ui_url:
            console.print("[yellow]Warning: --ui-url overrides --ui[/]")

        url_to_use = ui_url

        # If --ui flag is used and no explicit URL, infer it
        if ui and not ui_url:
            try:
                version = get_package_version("nodetool-core")
                # Fix normalized version if needed (e.g., 0.6.3rc12 -> 0.6.3-rc.12)
                if "rc" in version and "-" not in version:
                    version = version.replace("rc", "-rc.")

                url_to_use = (
                    f"https://github.com/nodetool-ai/nodetool/releases/download/v{version}/nodetool-web-{version}.zip"
                )
                console.print(f"[cyan]Inferring UI URL for version {version}: {url_to_use}[/]")
            except PackageNotFoundError:
                console.print("[red]Could not determine package version.[/]")
                # We will try fallback below if we can't infer or if inference fails
                pass

        if static_folder:
            console.print("[yellow]Warning: --ui-url/--ui overrides --static-folder[/]")

        try:
            if url_to_use:
                static_folder = download_and_cache_ui(url_to_use)
            else:
                raise Exception("No URL available")
        except Exception as e:
            if ui and not ui_url:
                console.print(f"[yellow]Failed to download UI for current version ({e}). Trying latest release...[/]")
                try:
                    import httpx

                    # Find latest tag
                    resp = httpx.get("https://github.com/nodetool-ai/nodetool/releases/latest", follow_redirects=False)
                    location = resp.headers.get("location", "")
                    if "/tag/" in location:
                        tag = location.split("/")[-1]
                        version_str = tag.lstrip("v")
                        latest_url = f"https://github.com/nodetool-ai/nodetool/releases/download/{tag}/nodetool-web-{version_str}.zip"
                        console.print(f"[cyan]Downloading latest UI ({tag}): {latest_url}[/]")
                        static_folder = download_and_cache_ui(latest_url)
                    else:
                        raise Exception("Could not resolve latest release tag")
                except Exception as inner_e:
                    console.print(f"[red]Failed to download latest UI: {inner_e}[/]")
                    sys.exit(1)
            else:
                console.print(f"[red]Failed to download UI: {e}[/]")
                sys.exit(1)
    return static_folder


def download_and_cache_ui(url: str) -> str:
    """Download UI zip from URL and unpack to cache directory."""
    import hashlib
    import shutil
    import zipfile
    from io import BytesIO

    import httpx

    from nodetool.config.settings import get_system_cache_path

    # Generate cache key from URL
    url_hash = hashlib.md5(url.encode()).hexdigest()
    cache_dir = get_system_cache_path("ui_cache") / url_hash

    # Check if already cached
    if cache_dir.exists() and (cache_dir / "index.html").exists():
        console.print(f"[green]Using cached UI from {cache_dir}[/]")
        return str(cache_dir)

    console.print(f"[cyan]Downloading UI from {url}...[/]")

    # Download
    with httpx.Client(follow_redirects=True) as client:
        resp = client.get(url)
        if resp.status_code == 404:
            raise Exception("404 Not Found")
        resp.raise_for_status()

        # Unpack
        console.print("[cyan]Unpacking UI...[/]")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(BytesIO(resp.content)) as z:
            z.extractall(cache_dir)

        # Handle case where zip contains a single top-level folder
        # e.g. nodetool-web-0.6.3-rc.12/index.html -> move contents up
        items = list(cache_dir.iterdir())
        if len(items) == 1 and items[0].is_dir():
            subdir = items[0]
            # Move contents to temp dir first to avoid conflicts
            temp_dir = cache_dir.parent / f"{url_hash}_temp"
            subdir.rename(temp_dir)
            try:
                shutil.rmtree(cache_dir)
                temp_dir.rename(cache_dir)
            except Exception:
                # Fallback cleanup
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise

        console.print(f"[green]UI ready at {cache_dir}[/]")
        return str(cache_dir)
