"""PDF tools module."""

import json
import os
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    import pymupdf
    import pymupdf4llm
    import pypandoc

from nodetool.workflows.processing_context import ProcessingContext

from .base import Tool


class ExtractPDFTextTool(Tool):
    name = "extract_pdf_text"
    description = "Extract plain text from a PDF document"
    input_schema: ClassVar[dict[str, Any]] = {
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
            import pymupdf

            path = context.resolve_workspace_path(params["path"])
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

    def user_message(self, params: dict) -> str:
        path = params.get("path", "a PDF")
        msg = f"Extracting text from {path}..."
        if len(msg) > 80:
            msg = "Extracting text from PDF..."
        return msg


class ExtractPDFTablesTool(Tool):
    name = "extract_pdf_tables"
    description = "Extract tables from a PDF document"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the PDF file",
            },
            "output_file": {
                "type": "string",
                "description": "Path to the output file",
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
            import pymupdf

            path = context.resolve_workspace_path(params["path"])
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
                            "external": (table.header.external if table.header else False),
                        },
                        "content": table.extract(),
                    }
                    all_tables.append(table_data)

            output_file = context.resolve_workspace_path(params["output_file"])
            with open(output_file, "w") as f:
                json.dump(all_tables, f)

            return {"output_file": output_file}
        except Exception as e:
            return {"error": str(e)}

    def user_message(self, params: dict) -> str:
        path = params.get("path", "a PDF")
        output = params.get("output_file", "output")
        msg = f"Extracting tables from {path} to {output}..."
        if len(msg) > 80:
            msg = f"Extracting tables from PDF to {output}..."
        if len(msg) > 80:
            msg = "Extracting tables from PDF..."
        return msg


class ConvertPDFToMarkdownTool(Tool):
    name = "convert_pdf_to_markdown"
    description = "Convert PDF to Markdown format"
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "input_file": {
                "type": "string",
                "description": "Path to the input PDF file",
            },
            "output_file": {
                "type": "string",
                "description": "Path to the output Markdown file",
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
        "required": ["input_file", "output_file"],
    }

    async def process(self, context: ProcessingContext, params: dict):
        try:
            import pymupdf
            import pymupdf4llm

            input_file = context.resolve_workspace_path(params["input_file"])
            output_file = context.resolve_workspace_path(params["output_file"])

            doc = pymupdf.open(input_file)

            md_text = pymupdf4llm.to_markdown(doc)

            # If page range is specified, split and extract relevant pages
            start_page = params.get("start_page", 0)
            end_page = params.get("end_page", -1)
            if start_page != 0 or end_page != -1:
                pages = md_text.split("\f")  # Split by form feed character
                end = end_page if end_page != -1 else len(pages) - 1
                md_text = "\f".join(pages[start_page : end + 1])

            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                f.write(md_text)

            return {"output_file": output_file}
        except Exception as e:
            return {"error": str(e)}

    def user_message(self, params: dict) -> str:
        input_file = params.get("input_file", "a PDF")
        output_file = params.get("output_file", "Markdown")
        msg = f"Converting {input_file} to {output_file}..."
        if len(msg) > 80:
            msg = f"Converting PDF to {output_file}..."
        if len(msg) > 80:
            msg = "Converting PDF to Markdown..."
        return msg


class ConvertMarkdownToPDFTool(Tool):
    name = "convert_markdown_to_pdf"
    description = "Convert Markdown to PDF using Pandoc."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "input_file": {
                "type": "string",
                "description": "Path to the input Markdown file",
            },
            "output_file": {
                "type": "string",
                "description": "Path to the output PDF file",
            },
        },
        "required": ["input_file", "output_file"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            import pypandoc

            input_file = context.resolve_workspace_path(params["input_file"])
            output_file = context.resolve_workspace_path(params["output_file"])

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            # Convert using pypandoc
            pypandoc.convert_file(
                str(input_file),
                "pdf",
                format=params.get("format_from", "markdown"),
                outputfile=str(output_file),
            )

            return {"output_file": output_file, "status": "success"}

        except Exception as e:
            return {"error": str(e)}

    def user_message(self, params: dict) -> str:
        input_file = params.get("input_file", "Markdown")
        output_file = params.get("output_file", "a PDF")
        msg = f"Converting {input_file} to {output_file}..."
        if len(msg) > 80:
            msg = f"Converting Markdown to {output_file}..."
        if len(msg) > 80:
            msg = "Converting Markdown to PDF..."
        return msg


class ConvertDocumentTool(Tool):
    name = "convert_document"
    description = "Convert between document formats using Pandoc, supports markdown, docx, rst, pdf, html, etc."
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "input_file": {
                "type": "string",
                "description": "Path to the input file",
            },
            "output_file": {
                "type": "string",
                "description": "Path to the output file",
            },
            "from_format": {
                "type": "string",
                "description": "Input format (e.g., markdown, docx, rst)",
                "default": "markdown",
            },
            "to_format": {
                "type": "string",
                "description": "Output format (e.g., pdf, docx, html)",
                "default": "pdf",
            },
            "extra_args": {
                "type": "array",
                "description": "Additional Pandoc arguments",
                "items": {"type": "string"},
                "default": [],
            },
        },
        "required": ["input_file", "output_file"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            import pypandoc

            input_file = context.resolve_workspace_path(params["input_file"])
            output_file = context.resolve_workspace_path(params["output_file"])

            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            extra_args = params.get("extra_args", [])
            if isinstance(extra_args, str):
                extra_args = [extra_args]

            # Convert using pypandoc
            pypandoc.convert_file(
                str(input_file),
                params.get("to_format", "pdf"),
                format=params.get("from_format", "markdown"),
                outputfile=str(output_file),
                extra_args=extra_args,
            )

            return {"output_file": output_file, "status": "success"}

        except Exception as e:
            return {"error": str(e)}

    def user_message(self, params: dict) -> str:
        input_file = params.get("input_file", "input")
        output_file = params.get("output_file", "output")
        to_format = params.get("to_format", "target format")
        msg = f"Converting {input_file} to {output_file} ({to_format})..."
        if len(msg) > 80:
            msg = f"Converting {input_file} to {to_format}..."
        if len(msg) > 80:
            msg = f"Converting document to {to_format}..."
        return msg
