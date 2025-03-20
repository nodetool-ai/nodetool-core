"""
Email tools module.

This module provides tools for working with email (Gmail):
- SearchEmailTool: Search Gmail messages
- ArchiveEmailTool: Archive Gmail messages
- AddLabelTool: Add labels to Gmail messages
"""

import imaplib
import html2text
import email
from email.header import decode_header
from datetime import datetime, timedelta
from typing import Any, Dict, Tuple

from nodetool.workflows.processing_context import ProcessingContext
from .base import Tool


def create_gmail_connection(
    context: ProcessingContext,
) -> tuple[imaplib.IMAP4_SSL, str, str]:
    """Helper function to create Gmail IMAP connection"""
    email_address = context.environment.get("GOOGLE_MAIL_USER")
    app_password = context.environment.get("GOOGLE_APP_PASSWORD")

    if not email_address:
        raise ValueError("GOOGLE_MAIL_USER is not set")
    if not app_password:
        raise ValueError("GOOGLE_APP_PASSWORD is not set")

    imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
    imap.login(email_address, app_password)

    return imap, email_address, app_password


def decode_bytes(byte_data: bytes, charset: str = "utf-8") -> str:
    """Helper function to decode bytes with multiple fallback options

    Args:
        byte_data: The bytes to decode
        charset: The initial charset to try (defaults to utf-8)

    Returns:
        Decoded string
    """
    # Guard against None bytes
    if byte_data is None:
        return ""

    # Try the specified charset first
    if charset:
        try:
            return byte_data.decode(charset)
        except (UnicodeDecodeError, LookupError):
            pass  # Continue to fallbacks

    # Try common encodings, including East Asian encodings
    encodings = [
        # Unicode encodings
        "utf-8",
        # Chinese encodings
        "gb2312",
        "gbk",
        "big5",
        "gb18030",
        # Japanese encodings
        "shift_jis",
        "euc-jp",
        "iso-2022-jp",
        # Korean encodings
        "euc-kr",
        "cp949",
        "iso-2022-kr",
        # Western encodings
        "latin1",
        "cp1252",
        "iso-8859-1",
        # Last resort
        "ascii",
    ]

    for encoding in encodings:
        try:
            return byte_data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue

    # Absolute last resort: use utf-8 with replacement for invalid chars
    return byte_data.decode("utf-8", errors="replace")


def parse_email_message(msg_data: tuple) -> Dict[str, Any]:
    """Helper function to parse email message data"""
    email_body = email.message_from_bytes(msg_data[0][1])

    # Decode subject
    subject = decode_header(email_body["subject"])[0]
    if isinstance(subject[0], bytes):
        # Try to decode with the specified charset, fall back to alternatives if that fails
        charset = subject[1] or "utf-8"
        subject_text = decode_bytes(subject[0], charset)
    else:
        subject_text = str(subject[0])

    # Get body content
    body = ""
    try:
        if email_body.is_multipart():
            for part in email_body.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True)
                    break
        else:
            body = email_body.get_payload(decode=True)

        # Decode body
        body = decode_bytes(body)  # type: ignore
    except Exception:
        body = ""

    return {
        "id": msg_data[0][0].decode(),
        "subject": subject_text,
        "from_address": email_body["from"],
        "to_address": email_body["to"],
        "date": email_body["date"],
        "body": body,
    }


class SearchEmailTool(Tool):
    name = "search_email"
    description = (
        "Search Gmail using various criteria and return subject, sender and message IDs"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "subject": {
                "type": "string",
                "description": "Text to search for in email subject",
            },
            "since_hours_ago": {
                "type": "integer",
                "description": "Number of hours ago to search for",
                "default": 6,
            },
            "text": {
                "type": "string",
                "description": "General text to search for anywhere in the email",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of emails to return",
                "default": 50,
            },
        },
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            imap, _, _ = create_gmail_connection(context)

            try:
                # Select folder
                imap.select("INBOX")

                # Build search criteria using Gmail's search syntax

                search_criteria = []

                if params.get("subject"):
                    search_criteria.append(f'SUBJECT "{params["subject"]}"')
                if params.get("text"):
                    search_criteria.append(f'BODY "{params["text"]}"')

                # Add date filter
                since_date = (
                    datetime.now()
                    - timedelta(hours=int(params.get("since_hours_ago", 6)))
                ).strftime("%d-%b-%Y")
                search_criteria.append(f"SINCE {since_date}")

                # Combine search criteria
                search_string = " ".join(search_criteria) if search_criteria else "ALL"

                # Perform search with UTF-8 encoding
                _, message_numbers = imap.uid("search", None, search_string)  # type: ignore

                if not message_numbers or not message_numbers[0]:
                    return {
                        "results": [],
                        "count": 0,
                        "message": "No emails found matching the criteria",
                    }

                email_ids = message_numbers[0].split()

                # Reverse the order to get newest first
                email_ids = list(reversed(email_ids))

                # Limit results
                max_results = min(len(email_ids), int(params.get("max_results") or 50))

                # Fetch only header information for each email
                detailed_results = []
                for i in range(max_results):
                    email_id = email_ids[i]
                    _, msg_data = imap.uid("fetch", email_id, "(RFC822)")
                    if msg_data and msg_data[0] is not None:
                        # Parse the header data
                        header_data = email.message_from_bytes(msg_data[0][1])
                        subject = decode_header(header_data["subject"])[0]
                        if isinstance(subject[0], bytes):
                            charset = subject[1] or "utf-8"
                            subject_text = decode_bytes(subject[0], charset)
                        else:
                            subject_text = str(subject[0]) if subject[0] else ""

                        body = ""
                        is_html = False
                        if header_data.is_multipart():
                            for part in header_data.walk():
                                body = part.get_payload(decode=True)
                                if part.get_content_type() == "text/html":
                                    is_html = True
                                break
                        else:
                            body = header_data.get_payload(decode=True)

                        if isinstance(body, bytes):
                            body = decode_bytes(body)  # type: ignore
                        elif isinstance(body, list):
                            body = "\n".join(decode_bytes(item) for item in body)

                        if is_html:
                            body = html2text.html2text(body)  # type: ignore

                        detailed_results.append(
                            {
                                "message_id": email_id.decode(),
                                "subject": subject_text,
                                "sender": header_data["from"],
                                "body": body,
                            }
                        )

                print(f"Found {len(detailed_results)} emails")
                print(f"Detailed results: {detailed_results}")
                return {"results": detailed_results, "count": len(detailed_results)}

            finally:
                imap.logout()

        except Exception as e:
            import traceback

            traceback.print_exc()
            return {"error": str(e)}


class ArchiveEmailTool(Tool):
    name = "archive_email"
    description = "Move specified emails to Gmail archive"
    input_schema = {
        "type": "object",
        "properties": {
            "message_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of message IDs to archive",
            },
        },
        "required": ["message_ids"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            imap, _, _ = create_gmail_connection(context)

            try:
                imap.select("INBOX")

                archived_ids = []
                for message_id in params["message_ids"]:
                    # Moving to archive in Gmail is done by removing the INBOX label
                    result = imap.store(message_id, "-X-GM-LABELS", "\\Inbox")
                    if result[0] == "OK":
                        archived_ids.append(message_id)

                return {
                    "success": True,
                    "archived_messages": archived_ids,
                }

            finally:
                imap.logout()

        except Exception as e:
            return {"error": str(e)}


class AddLabelTool(Tool):
    name = "add_label"
    description = "Add a label to a Gmail message"
    input_schema = {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "Message ID to label",
            },
            "label": {
                "type": "string",
                "description": "Label to add to the message",
            },
        },
        "required": ["message_id", "label"],
    }

    async def process(self, context: ProcessingContext, params: dict) -> Any:
        try:
            imap, _, _ = create_gmail_connection(context)

            try:
                imap.select("INBOX")

                result = imap.store(
                    params["message_id"], "+X-GM-LABELS", params["label"]
                )

                return {
                    "success": result[0] == "OK",
                    "message_id": params["message_id"],
                    "label": params["label"],
                }

            finally:
                imap.logout()

        except Exception as e:
            return {"error": str(e)}
