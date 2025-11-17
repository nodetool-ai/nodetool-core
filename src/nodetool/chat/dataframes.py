from typing import Any, ClassVar, Dict

from nodetool.agents.tools.base import Tool
from nodetool.metadata.types import ColumnDef, RecordType
from nodetool.workflows.processing_context import ProcessingContext


def json_schema_for_column(column: ColumnDef) -> dict:
    """Create a JSON schema for a database column definition.

    Converts a ColumnDef object to a JSON schema representation that can be used in JSON schema
    validation. Different data types are mapped to appropriate JSON schema types with format
    specifications where applicable.

    Args:
        column (ColumnDef): The column definition containing name, data type, and description

    Returns:
        dict: A JSON schema object representing the column with type and description

    Raises:
        ValueError: If an unsupported data type is encountered
    """
    data_type = column.data_type
    description = column.description or ""

    if data_type == "string":
        return {"type": "string", "description": description}
    if data_type == "number":
        return {"type": "number", "description": description}
    if data_type == "int":
        return {"type": "integer", "description": description}
    if data_type == "float":
        return {"type": "number", "description": description}
    if data_type == "datetime":
        return {"type": "string", "format": "date-time", "description": description}
    raise ValueError(f"Unknown data type {data_type}")


def json_schema_for_dictionary(fields: RecordType) -> dict:
    """Create a JSON schema for a dictionary.

    Converts a RecordType object to a JSON schema representation that can be used in JSON schema
    validation.
    """
    return {
        "type": "object",
        "properties": {
            field.name: json_schema_for_column(field) for field in fields.columns
        },
        "required": [field.name for field in fields.columns],
        "additionalProperties": False,
    }


class GenerateStringTool(Tool):
    name: str = "generate_string"
    input_schema: ClassVar[dict] = {
        "type": "object",
        "properties": {
            "string": {
                "type": "string",
                "description": "The generated string",
            }
        },
        "required": ["string"],
        "additionalProperties": False,
    }

    def __init__(self, description: str):
        self.description = description

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        return params


class GenerateDataTool(Tool):
    name: str = "create_record"

    def __init__(self, description: str, columns: list[ColumnDef]):
        self.description = description
        self.columns = columns
        self.input_schema = {
            "type": "object",
            "properties": {
                column.name: json_schema_for_column(column) for column in columns
            },
            "required": [column.name for column in columns],
            "additionalProperties": False,
        }

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        return params


def json_schema_for_dataframe(columns: list[ColumnDef]) -> dict:
    """Create a JSON schema for a DataFrame.

    Builds a comprehensive JSON schema that represents a DataFrame structure with
    the specified columns. The schema enforces that all required columns are present
    and prevents additional properties.

    Args:
        columns (list[ColumnDef]): List of column definitions for the DataFrame

    Returns:
        dict: A JSON schema object with nested properties representing the DataFrame structure
            with a "data" array containing objects that conform to the column definitions
    """
    return {
        "type": "object",
        "properties": {
            "data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        column.name: json_schema_for_column(column)
                        for column in columns
                    },
                    "required": [column.name for column in columns],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["data"],
        "additionalProperties": False,
    }
