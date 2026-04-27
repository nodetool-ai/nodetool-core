import pandas as pd
import pytest

from nodetool.metadata.types import ColumnDef, DataframeRef
from nodetool.workflows.processing_context import ProcessingContext


@pytest.mark.asyncio
async def test_dataframe_memory_uri_roundtrip(context: ProcessingContext):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    # Convert to DataframeRef using in-memory URI
    ref = await context.dataframe_from_pandas(df)

    assert isinstance(ref, DataframeRef)
    assert ref.uri.startswith("memory://")
    assert ref.asset_id is None
    # Pure reference should not carry materialized data/columns
    assert ref.columns is None and ref.data is None

    # Convert back to pandas and verify equality
    df_roundtrip = await context.dataframe_to_pandas(ref)
    assert list(df_roundtrip.columns) == ["a", "b"]
    assert df_roundtrip.equals(df)


@pytest.mark.asyncio
async def test_dataframe_from_columns_and_data_fallback(context: ProcessingContext):
    # Fallback path: DataframeRef with columns/data should still convert
    ref = DataframeRef(
        columns=[
            ColumnDef(name="a", data_type="int"),
            ColumnDef(name="b", data_type="string"),
        ],
        data=[[1, "x"], [2, "y"]],
    )

    df = await context.dataframe_to_pandas(ref)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


@pytest.mark.asyncio
async def test_dataframe_from_json_asset(context: ProcessingContext):
    # Test loading from JSON asset (external file scenario)
    import json
    import tempfile
    from pathlib import Path

    df_data = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json", dir=context.workspace_dir) as f:
        json.dump(df_data, f)
        filepath = Path(f.name)

    try:
        # Create DataframeRef pointing to the file, without columns/data
        # This forces the else block in dataframe_to_pandas
        # Use relative path since download_file expects workspace-relative path handling for traversal prevention
        # file:///path is needed instead of file://name to make scheme parser happy, or use proper relative
        # Actually, let's use the absolute path since resolve_workspace_path handles absolute paths
        # by checking if they are within the workspace.
        # But wait, resolve_workspace_path strips the leading slash and treats it as relative
        # to workspace. So we should pass the relative path.
        # And for `file://` scheme properly parsing it, we might need a slash to avoid it becoming netloc
        relative_path = filepath.relative_to(context.workspace_dir)
        ref = DataframeRef(uri=f"file:///{relative_path}")

        df = await context.dataframe_to_pandas(ref)

        assert list(df.columns) == ["a", "b"]
        assert df.shape == (2, 2)
        assert df.iloc[0]["a"] == 1
        assert df.iloc[0]["b"] == "x"
    finally:
        filepath.unlink()
