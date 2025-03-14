"""
PDF tools module.

This module provides tools for working with PDF documents:
- ExtractPDFTextTool: Extract text from PDFs
- ExtractPDFTablesTool: Extract tables from PDFs
- ConvertPDFToMarkdownTool: Convert PDFs to markdown
"""

import os
from typing import Any

import pymupdf
import pymupdf4llm

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


class ExtractPDFTextTool(Tool):
    def __init__(self):
        super().__init__(
            name="extract_pdf_text",
            description="Extract plain text from a PDF document",
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file",
                },
                "start_page": {
                    "type": "integer",
                    "description": "First page to extract (0-based index)",
                    "default": 0,
                },
                "end_page": {
                    "type": "integer",
                    "description": "Last page to extract (-1 for last page)",
                    "default": -1,
                },
            },
            "required": ["path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            path = os.path.expanduser(params["path"])
            doc = pymupdf.open(path)

            end = params.get("end_page", -1)
            if end == -1:
                end = doc.page_count - 1

            text = ""
            for page_num in range(params.get("start_page", 0), end + 1):
                page = doc[page_num]
                text += page.get_text()  # type: ignore

            return {"text": text}
        except Exception as e:
            return {"error": str(e)}


class ExtractPDFTablesTool(Tool):
    def __init__(self):
        super().__init__(
            name="extract_pdf_tables", description="Extract tables from a PDF document"
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file",
                },
                "start_page": {
                    "type": "integer",
                    "description": "First page to extract (0-based index)",
                    "default": 0,
                },
                "end_page": {
                    "type": "integer",
                    "description": "Last page to extract (-1 for last page)",
                    "default": -1,
                },
            },
            "required": ["path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            path = os.path.expanduser(params["path"])
            doc = pymupdf.open(path)

            end = params.get("end_page", -1)
            if end == -1:
                end = doc.page_count - 1

            all_tables = []
            for page_num in range(params.get("start_page", 0), end + 1):
                page = doc[page_num]
                tables = page.find_tables()  # type: ignore

                for table in tables:
                    table_data = {
                        "page": page_num,
                        "bbox": {
                            "x0": table.bbox[0],
                            "y0": table.bbox[1],
                            "x1": table.bbox[2],
                            "y1": table.bbox[3],
                        },
                        "rows": table.row_count,
                        "columns": table.col_count,
                        "header": {
                            "names": table.header.names if table.header else [],
                            "external": (
                                table.header.external if table.header else False
                            ),
                        },
                        "content": table.extract(),
                    }
                    all_tables.append(table_data)

            return {"tables": all_tables}
        except Exception as e:
            return {"error": str(e)}


class ConvertPDFToMarkdownTool(Tool):
    def __init__(self):
        super().__init__(
            name="convert_pdf_to_markdown", description="Convert PDF to Markdown format"
        )
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the PDF file",
                },
                "start_page": {
                    "type": "integer",
                    "description": "First page to extract (0-based index)",
                    "default": 0,
                },
                "end_page": {
                    "type": "integer",
                    "description": "Last page to extract (-1 for last page)",
                    "default": -1,
                },
            },
            "required": ["path"],
        }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            path = os.path.expanduser(params["path"])
            doc = pymupdf.open(path)

            md_text = pymupdf4llm.to_markdown(doc)

            # If page range is specified, split and extract relevant pages
            start_page = params.get("start_page", 0)
            end_page = params.get("end_page", -1)
            if start_page != 0 or end_page != -1:
                pages = md_text.split("\f")  # Split by form feed character
                end = end_page if end_page != -1 else len(pages) - 1
                md_text = "\f".join(pages[start_page : end + 1])

            return {"markdown": md_text}
        except Exception as e:
            return {"error": str(e)}
