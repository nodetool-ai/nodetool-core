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
