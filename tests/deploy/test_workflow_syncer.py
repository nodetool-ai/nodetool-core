from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nodetool.deploy.admin_client import AdminHTTPClient
from nodetool.deploy.workflow_syncer import WorkflowSyncer


@pytest.mark.asyncio
async def test_sync_workflow_success():
    # Mock client
    client = AsyncMock(spec=AdminHTTPClient)
    client.update_workflow.return_value = {"status": "ok"}

    async def mock_download_gen(*args, **kwargs):
        yield {"status": "complete", "file": "test"}

    client.download_huggingface_model.side_effect = mock_download_gen
    client.download_ollama_model.side_effect = mock_download_gen

    client.get_asset.side_effect = Exception("Not found") # To trigger create
    client.create_asset.return_value = {}
    client.upload_asset_file.return_value = {}

    # Mock console
    console = MagicMock()

    # Mock Workflow.get
    with patch("nodetool.models.workflow.Workflow.get", new_callable=AsyncMock) as mock_get_workflow:
        workflow = MagicMock()
        # Mocking what from_model returns (Pydantic model)
        workflow_pydantic = MagicMock()
        workflow_pydantic.model_dump.return_value = {
            "id": "wf1",
            "graph": {
                "nodes": [
                    {
                        "type": "nodetool.constant.string",
                        "data": {"value": {"asset_id": "asset1"}}
                    }
                ]
            }
        }
        mock_get_workflow.return_value = workflow

        # Mock from_model
        with patch("nodetool.deploy.workflow_syncer.from_model", new_callable=AsyncMock) as mock_from_model:
            mock_from_model.return_value = workflow_pydantic

            # Mock extract_models
            with patch("nodetool.deploy.workflow_syncer.extract_models") as mock_extract_models:
                mock_extract_models.return_value = [
                    {"type": "hf.model", "repo_id": "test/model", "path": "model.safetensors"}
                ]

                # Mock AssetModel.get
                with patch("nodetool.models.asset.Asset.get", new_callable=AsyncMock) as mock_get_asset:
                    asset = MagicMock()
                    asset.id = "asset1"
                    asset.user_id = "user1"
                    asset.name = "test_asset"
                    asset.content_type = "image/png"
                    asset.file_name = "test.png"
                    asset.has_thumbnail = False
                    mock_get_asset.return_value = asset

                    # Mock require_scope
                    with patch("nodetool.deploy.workflow_syncer.require_scope") as mock_require_scope:
                        storage = AsyncMock()
                        storage.download.return_value = None # Should write to stream

                        # Mock BytesIO usage in download
                        async def mock_download(filename, stream):
                            stream.write(b"fake data")

                        storage.download.side_effect = mock_download
                        mock_require_scope.return_value.get_asset_storage.return_value = storage

                        syncer = WorkflowSyncer(client, console)
                        result = await syncer.sync_workflow("wf1")

                        assert result is True
                        client.update_workflow.assert_called_once()
                        client.create_asset.assert_called_once()
                        client.upload_asset_file.assert_called_once()
                        client.download_huggingface_model.assert_called_once()

@pytest.mark.asyncio
async def test_sync_workflow_not_found():
    client = AsyncMock(spec=AdminHTTPClient)
    console = MagicMock()

    with patch("nodetool.models.workflow.Workflow.get", new_callable=AsyncMock) as mock_get_workflow:
        mock_get_workflow.return_value = None

        syncer = WorkflowSyncer(client, console)
        result = await syncer.sync_workflow("wf1")

        assert result is False
        console.print.assert_called_with("[red]❌ Workflow not found locally: wf1[/]")

@pytest.mark.asyncio
async def test_sync_workflow_exception():
    client = AsyncMock(spec=AdminHTTPClient)
    console = MagicMock()

    with patch("nodetool.models.workflow.Workflow.get", new_callable=AsyncMock) as mock_get_workflow:
        mock_get_workflow.side_effect = Exception("DB Error")

        syncer = WorkflowSyncer(client, console)
        result = await syncer.sync_workflow("wf1")

        assert result is False
        console.print.assert_called()
