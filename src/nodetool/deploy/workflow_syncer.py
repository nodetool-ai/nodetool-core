import asyncio
from io import BytesIO
from typing import Optional

from rich.console import Console

from nodetool.api.workflow import from_model
from nodetool.deploy.admin_client import AdminHTTPClient
from nodetool.deploy.sync import extract_models
from nodetool.models.asset import Asset as AssetModel
from nodetool.models.workflow import Workflow
from nodetool.runtime.resources import require_scope


class WorkflowSyncer:
    def __init__(self, client: AdminHTTPClient, console: Optional[Console] = None):
        self.client = client
        self.console = console or Console()

    async def sync_workflow(self, workflow_id: str) -> bool:
        """Sync a workflow, its assets, and required models to the remote instance.

        Args:
            workflow_id: The ID of the workflow to sync.

        Returns:
            bool: True if sync was successful, False otherwise.
        """
        try:
            # Get local workflow
            workflow = await Workflow.get(workflow_id)
            if workflow is None:
                self.console.print(f"[red]❌ Workflow not found locally: {workflow_id}[/]")
                return False

            # Sync assets first
            workflow_data = (await from_model(workflow)).model_dump()
            synced_assets = await self._extract_and_sync_assets(workflow_data)
            if synced_assets > 0:
                self.console.print(f"[green]✅ Synced {synced_assets} asset(s)[/]")
                self.console.print()

            # Download models required by the workflow
            synced_models = await self._extract_and_download_models(workflow_data)
            if synced_models > 0:
                self.console.print(f"[green]✅ Downloaded {synced_models} model(s)[/]")
                self.console.print()

            # Sync workflow
            await self.client.update_workflow(workflow_id, workflow_data)
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Failed to sync workflow: {e}[/]")
            import traceback

            traceback.print_exc()
            return False

    async def _extract_and_download_models(self, workflow_data: dict) -> int:
        """Extract model references from workflow and download them on remote."""
        models = extract_models(workflow_data)

        if not models:
            return 0

        self.console.print(f"[cyan]Found {len(models)} model(s) to download[/]")

        downloaded_count = 0
        for model in models:
            try:
                model_type = model.get("type", "")

                # Handle HuggingFace models
                if model_type.startswith("hf."):
                    repo_id = model.get("repo_id")
                    if not repo_id:
                        self.console.print("  [red]Error: repo_id is required for HF models[/]")
                        continue
                    self.console.print(f"  [cyan]Downloading HF model: {repo_id}[/]")

                    # Start download (streaming progress)
                    last_status = None
                    async for progress in self.client.download_huggingface_model(
                        repo_id=repo_id,  # type: ignore[arg-type]
                        file_path=model.get("path"),
                        ignore_patterns=model.get("ignore_patterns"),
                        allow_patterns=model.get("allow_patterns"),
                    ):
                        last_status = progress.get("status")
                        if last_status == "downloading":
                            file_name = progress.get("file", "")
                            percent = progress.get("percent", 0)
                            self.console.print(
                                f"    [yellow]{file_name}: {percent:.1f}%[/]",
                                end="\r",
                            )
                        elif last_status == "complete":
                            self.console.print(f"    [green]✓ Downloaded {repo_id}[/]")

                    # Stream completed - mark as downloaded
                    if last_status != "complete":
                        self.console.print(f"    [green]✓ Downloaded {repo_id}[/]")
                    downloaded_count += 1

                # Handle Ollama models
                elif model_type == "language_model" and model.get("provider") == "ollama":
                    model_id = model.get("id")
                    if not model_id:
                        self.console.print("  [red]Error: model id is required for Ollama models[/]")
                        continue
                    self.console.print(f"  [cyan]Downloading Ollama model: {model_id}[/]")

                    last_status = None
                    async for progress in self.client.download_ollama_model(model_name=model_id):  # type: ignore[arg-type]
                        last_status = progress.get("status")
                        if last_status and last_status != "success":
                            self.console.print(f"    [yellow]{last_status}[/]", end="\r")
                        elif last_status == "success":
                            self.console.print(f"    [green]✓ Downloaded {model_id}[/]")

                    # Stream completed - mark as downloaded
                    if last_status != "success":
                        self.console.print(f"    [green]✓ Downloaded {model_id}[/]")
                    downloaded_count += 1

            except Exception as e:
                self.console.print(f"    [red]✗ Failed to download model: {e}[/]")

        return downloaded_count

    async def _extract_and_sync_assets(self, workflow_data: dict) -> int:
        """Extract asset references from workflow and sync them to remote."""
        asset_ids = set()

        # Extract asset IDs from constant nodes
        for node in workflow_data.get("graph", {}).get("nodes", []):
            node_type = node.get("type", "")
            if node_type.startswith("nodetool.constant."):
                value = node.get("data", {}).get("value", {})
                if isinstance(value, dict):
                    # Check for asset_id field
                    asset_id = value.get("asset_id")
                    if asset_id:
                        asset_ids.add(asset_id)

        if not asset_ids:
            return 0

        self.console.print(f"[cyan]Found {len(asset_ids)} asset(s) to sync[/]")

        # Get local storage
        storage = require_scope().get_asset_storage()
        synced_count = 0

        for asset_id in asset_ids:
            try:
                # Get local asset metadata
                asset = await AssetModel.get(asset_id)
                if not asset:
                    self.console.print(f"  [yellow]⚠️  Asset {asset_id} not found locally, skipping[/]")
                    continue

                self.console.print(f"  [cyan]Syncing asset: {asset.name}[/]")

                # Check if asset already exists on remote
                try:
                    await self.client.get_asset(asset_id)
                    self.console.print("    [yellow]Asset already exists on remote, skipping[/]")
                    synced_count += 1
                    continue
                except Exception:
                    # Asset doesn't exist, continue with sync
                    pass

                # Create asset metadata on remote (preserve asset ID)
                await self.client.create_asset(
                    id=asset.id,
                    user_id=asset.user_id,
                    name=asset.name,
                    content_type=asset.content_type,
                    parent_id=asset.parent_id,
                    workflow_id=asset.workflow_id,
                    metadata=asset.metadata,
                )

                # Upload asset file if it's not a folder
                if asset.content_type != "folder" and asset.file_name:
                    # Download from local storage
                    stream = BytesIO()
                    await storage.download(asset.file_name, stream)
                    file_data = stream.getvalue()

                    # Upload to remote storage
                    await self.client.upload_asset_file(asset.file_name, file_data)

                    # Upload thumbnail if exists
                    if asset.has_thumbnail and asset.thumb_file_name:
                        thumb_stream = BytesIO()
                        await storage.download(asset.thumb_file_name, thumb_stream)
                        thumb_data = thumb_stream.getvalue()
                        await self.client.upload_asset_file(asset.thumb_file_name, thumb_data)

                self.console.print(f"    [green]✓ Synced {asset.name}[/]")
                synced_count += 1

            except Exception as e:
                self.console.print(f"    [red]✗ Failed to sync asset {asset_id}: {e}[/]")

        return synced_count
