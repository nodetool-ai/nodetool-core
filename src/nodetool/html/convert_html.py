from bs4 import BeautifulSoup
import re


def convert_html_to_text(html: str, preserve_linebreaks: bool = True) -> str:
    """
    Converts HTML to plain text while preserving structure and handling whitespace.

    Args:
        html: HTML string to convert
        preserve_linebreaks: Whether to preserve line breaks from block elements

    Returns:
        Cleaned plain text string
    """
    # Parse HTML with BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")

    # Handle line breaks if preserve_linebreaks is True
    if preserve_linebreaks:
        # Replace <br> tags with newlines
        for br in soup.find_all("br"):
            br.replace_with("\n")

        # Add newlines after block-level elements
        for tag in soup.find_all(
            ["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li"]
        ):
            tag.append("\n")

    # Get text content
    text = soup.get_text()

    # Clean up whitespace
    text = re.sub(
        r"\n\s*\n", "\n\n", text
    )  # Convert multiple blank lines to double line breaks
    text = re.sub(r" +", " ", text)  # Remove multiple spaces
    return text.strip()
