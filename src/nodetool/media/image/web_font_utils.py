"""
Web font utilities for downloading and caching fonts from Google Fonts and custom URLs.
"""

import hashlib
import os
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nodetool.config.logging_config import get_logger

log = get_logger(__name__)

# Google Fonts API base URL for raw TTF files
GOOGLE_FONTS_RAW_URL = "https://raw.githubusercontent.com/google/fonts/main"

# Cache directory for downloaded web fonts
_FONT_CACHE_DIR: Path | None = None


def get_font_cache_dir() -> Path:
    """Get the directory for caching downloaded web fonts.

    Returns:
        Path to the font cache directory
    """
    global _FONT_CACHE_DIR
    if _FONT_CACHE_DIR is None:
        # Use OS-appropriate cache directory
        home = Path.home()
        if os.name == "nt":  # Windows
            cache_base = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        elif os.name == "posix":
            if os.uname().sysname == "Darwin":  # macOS
                cache_base = home / "Library" / "Caches"
            else:  # Linux and other Unix
                cache_base = Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
        else:
            cache_base = home / ".cache"

        _FONT_CACHE_DIR = cache_base / "nodetool" / "fonts"
        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    return _FONT_CACHE_DIR


# Mapping of font family names to their directory structure in Google Fonts repo
# This is a subset of popular fonts - the full list can be fetched from the API
GOOGLE_FONTS_CATALOG = {
    # Sans-serif fonts
    "roboto": ("ofl/roboto", "Roboto"),
    "open sans": ("ofl/opensans", "OpenSans"),
    "opensans": ("ofl/opensans", "OpenSans"),
    "lato": ("ofl/lato", "Lato"),
    "montserrat": ("ofl/montserrat", "Montserrat"),
    "poppins": ("ofl/poppins", "Poppins"),
    "inter": ("ofl/inter", "Inter"),
    "nunito": ("ofl/nunito", "Nunito"),
    "raleway": ("ofl/raleway", "Raleway"),
    "ubuntu": ("ufl/ubuntu", "Ubuntu"),
    "work sans": ("ofl/worksans", "WorkSans"),
    "worksans": ("ofl/worksans", "WorkSans"),
    "source sans pro": ("ofl/sourcesans3", "SourceSans3"),
    "sourcesanspro": ("ofl/sourcesans3", "SourceSans3"),
    "oswald": ("ofl/oswald", "Oswald"),
    "pt sans": ("ofl/ptsans", "PTSans"),
    "ptsans": ("ofl/ptsans", "PTSans"),
    "noto sans": ("ofl/notosans", "NotoSans"),
    "notosans": ("ofl/notosans", "NotoSans"),
    "fira sans": ("ofl/firasans", "FiraSans"),
    "firasans": ("ofl/firasans", "FiraSans"),
    # Serif fonts
    "playfair display": ("ofl/playfairdisplay", "PlayfairDisplay"),
    "playfairdisplay": ("ofl/playfairdisplay", "PlayfairDisplay"),
    "merriweather": ("ofl/merriweather", "Merriweather"),
    "lora": ("ofl/lora", "Lora"),
    "pt serif": ("ofl/ptserif", "PTSerif"),
    "ptserif": ("ofl/ptserif", "PTSerif"),
    "source serif pro": ("ofl/sourceserifpro", "SourceSerifPro"),
    "sourceserifpro": ("ofl/sourceserifpro", "SourceSerifPro"),
    "noto serif": ("ofl/notoserif", "NotoSerif"),
    "notoserif": ("ofl/notoserif", "NotoSerif"),
    "libre baskerville": ("ofl/librebaskerville", "LibreBaskerville"),
    "librebaskerville": ("ofl/librebaskerville", "LibreBaskerville"),
    # Display fonts
    "bebas neue": ("ofl/bebasneue", "BebasNeue"),
    "bebasneue": ("ofl/bebasneue", "BebasNeue"),
    "abril fatface": ("ofl/abrilfatface", "AbrilFatface"),
    "abrilfatface": ("ofl/abrilfatface", "AbrilFatface"),
    "anton": ("ofl/anton", "Anton"),
    "righteous": ("ofl/righteous", "Righteous"),
    "lobster": ("ofl/lobster", "Lobster"),
    "pacifico": ("ofl/pacifico", "Pacifico"),
    "comfortaa": ("ofl/comfortaa", "Comfortaa"),
    "permanent marker": ("ofl/permanentmarker", "PermanentMarker"),
    "permanentmarker": ("ofl/permanentmarker", "PermanentMarker"),
    "orbitron": ("ofl/orbitron", "Orbitron"),
    "monoton": ("ofl/monoton", "Monoton"),
    "poiret one": ("ofl/poiretone", "PoiretOne"),
    # Monospace fonts
    "roboto mono": ("ofl/robotomono", "RobotoMono"),
    "robotomono": ("ofl/robotomono", "RobotoMono"),
    "source code pro": ("ofl/sourcecodepro", "SourceCodePro"),
    "sourcecodepro": ("ofl/sourcecodepro", "SourceCodePro"),
    "fira code": ("ofl/firacode", "FiraCode"),
    "firacode": ("ofl/firacode", "FiraCode"),
    "jetbrains mono": ("ofl/jetbrainsmono", "JetBrainsMono"),
    "jetbrainsmono": ("ofl/jetbrainsmono", "JetBrainsMono"),
    "space mono": ("ofl/spacemono", "SpaceMono"),
    "spacemono": ("ofl/spacemono", "SpaceMono"),
    "ubuntu mono": ("ufl/ubuntumono", "UbuntuMono"),
    "ubuntumono": ("ufl/ubuntumono", "UbuntuMono"),
    "inconsolata": ("ofl/inconsolata", "Inconsolata"),
    # Handwriting fonts
    "dancing script": ("ofl/dancingscript", "DancingScript"),
    "dancingscript": ("ofl/dancingscript", "DancingScript"),
    "caveat": ("ofl/caveat", "Caveat"),
    "shadows into light": ("ofl/shadowsintolight", "ShadowsIntoLight"),
    "shadowsintolight": ("ofl/shadowsintolight", "ShadowsIntoLight"),
    "indie flower": ("ofl/indieflower", "IndieFlower"),
    "indieflower": ("ofl/indieflower", "IndieFlower"),
    "satisfy": ("ofl/satisfy", "Satisfy"),
    "great vibes": ("ofl/greatvibes", "GreatVibes"),
    "greatvibes": ("ofl/greatvibes", "GreatVibes"),
    "sacramento": ("ofl/sacramento", "Sacramento"),
}

# Weight mapping for Google Fonts
WEIGHT_MAP = {
    "thin": "100",
    "extralight": "200",
    "light": "300",
    "regular": "400",
    "medium": "500",
    "semibold": "600",
    "bold": "700",
    "extrabold": "800",
    "black": "900",
    # Variants
    "normal": "400",
    "": "400",
    # Italic variants (using suffix)
    "italic": "400italic",
    "thinitalic": "100italic",
    "lightitalic": "300italic",
    "mediumitalic": "500italic",
    "semibolditalic": "600italic",
    "bolditalic": "700italic",
    "extrabolditalic": "800italic",
    "blackitalic": "900italic",
}


def _get_cache_filename(font_name: str, weight: str, url: str = "") -> str:
    """Generate a cache filename for a font.

    Args:
        font_name: Name of the font
        weight: Font weight
        url: Optional URL (for URL-based fonts)

    Returns:
        Cache filename
    """
    if url:
        # For URL-based fonts, use a hash of the URL
        from urllib.parse import urlsplit

        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        url_path = urlsplit(url).path
        extension = Path(url_path).suffix.lower()
        if extension not in {".ttf", ".otf"}:
            extension = ".ttf"
        return f"url_{url_hash}{extension}"
    else:
        # For Google Fonts, use font name and weight
        clean_name = re.sub(r"[^a-zA-Z0-9]", "", font_name.lower())
        return f"google_{clean_name}_{weight}.ttf"


def download_google_font(font_name: str, weight: str = "regular") -> str:
    """Download a font from Google Fonts and cache it locally.

    Args:
        font_name: Name of the Google Font (e.g., "Roboto", "Open Sans")
        weight: Font weight (e.g., "regular", "bold", "300", "700italic")

    Returns:
        Path to the downloaded font file

    Raises:
        ValueError: If the font is not found in the catalog
        ConnectionError: If the download fails
    """
    from urllib.parse import quote

    cache_dir = get_font_cache_dir()
    cache_filename = _get_cache_filename(font_name, weight)
    cache_path = cache_dir / cache_filename

    # Return cached file if it exists
    if cache_path.exists():
        log.debug(f"Using cached font: {cache_path}")
        return str(cache_path)

    # Look up font in catalog
    font_key = font_name.lower().strip()
    if font_key not in GOOGLE_FONTS_CATALOG:
        available = ", ".join(sorted({k for k in GOOGLE_FONTS_CATALOG if " " in k or k == font_key.replace(" ", "")}))
        raise ValueError(
            f"Font '{font_name}' not found in Google Fonts catalog. "
            f"Available fonts include: {available[:200]}... "
            f"For other fonts, use source='url' with a direct TTF URL."
        )

    font_dir, font_base = GOOGLE_FONTS_CATALOG[font_key]

    # Normalize weight
    weight_normalized = weight.lower().strip()
    weight_value = WEIGHT_MAP.get(weight_normalized, weight_normalized)

    # Handle italic
    is_italic = "italic" in weight_value.lower()
    weight_num = weight_value.replace("italic", "").strip()
    if not weight_num:
        weight_num = "400"

    # Construct filename patterns to try
    # Google Fonts uses various naming conventions with variable fonts
    # Filenames may include: [wght], [wdth,wght], -Italic[wght], etc.
    filename_patterns = []

    if is_italic:
        # Italic variants - try various variable font naming patterns
        filename_patterns.extend(
            [
                f"{font_base}-Italic[wdth,wght].ttf",  # Variable with width and weight
                f"{font_base}-Italic[wght].ttf",  # Variable with just weight
                f"{font_base}[wdth,wght].ttf",  # Variable font with both axes (may contain italic)
                f"{font_base}[wght].ttf",  # Variable font may contain italic
                f"{font_base}-Italic.ttf",  # Static italic
                f"{font_base}Italic-{weight_num}.ttf",
                f"{font_base}-{weight_num}italic.ttf",
            ]
        )
    else:
        # Regular variants - try various variable font naming patterns
        filename_patterns.extend(
            [
                f"{font_base}[wdth,wght].ttf",  # Variable with width and weight (like Roboto)
                f"{font_base}[wght].ttf",  # Variable with just weight
                f"{font_base}-VariableFont_wght.ttf",  # Variable font alternate naming
                f"{font_base}-Regular.ttf",  # Static regular
                f"{font_base}-{weight_num}.ttf",  # Static weight
                f"{font_base}.ttf",  # Simple naming
            ]
        )

    # Try each pattern
    for filename in filename_patterns:
        # URL-encode special characters in filenames (brackets, commas)
        encoded_filename = quote(filename, safe="")
        url = f"{GOOGLE_FONTS_RAW_URL}/{font_dir}/{encoded_filename}"
        log.debug(f"Trying to download font from: {url}")

        try:
            request = Request(url, headers={"User-Agent": "NodeTool/1.0"})
            with urlopen(request, timeout=30) as response:
                font_data = response.read()

            # Save to cache
            cache_path.write_bytes(font_data)
            log.info(f"Downloaded and cached Google Font: {font_name} ({weight}) -> {cache_path}")
            return str(cache_path)

        except HTTPError as e:
            if e.code == 404:
                log.debug(f"Font file not found at {url}, trying next pattern...")
                continue
            raise ConnectionError(f"Failed to download font from {url}: HTTP {e.code}") from e
        except URLError as e:
            raise ConnectionError(f"Failed to connect to download font: {e.reason}") from e

    raise ValueError(
        f"Could not find font file for '{font_name}' weight '{weight}' in Google Fonts. "
        f"Try a different weight (regular, bold, light, etc.) or use source='url' with a direct TTF URL."
    )


def download_font_from_url(url: str) -> str:
    """Download a font from a custom URL and cache it locally.

    Args:
        url: URL to the TTF/OTF font file

    Returns:
        Path to the downloaded font file

    Raises:
        ValueError: If the URL is invalid
        ConnectionError: If the download fails
    """
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"Invalid font URL: {url}. Must be http:// or https://")

    cache_dir = get_font_cache_dir()
    cache_filename = _get_cache_filename("", "", url)
    cache_path = cache_dir / cache_filename

    # Return cached file if it exists
    if cache_path.exists():
        log.debug(f"Using cached font from URL: {cache_path}")
        return str(cache_path)

    log.info(f"Downloading font from URL: {url}")

    try:
        request = Request(url, headers={"User-Agent": "NodeTool/1.0"})
        with urlopen(request, timeout=30) as response:
            font_data = response.read()

        # Basic validation - check for TTF/OTF magic bytes
        if len(font_data) < 4:
            raise ValueError("Downloaded file is too small to be a valid font")

        # TTF starts with 0x00010000 or 'OTTO' (for OTF) or 'true' or 'typ1'
        magic = font_data[:4]
        valid_signatures = [
            b"\x00\x01\x00\x00",  # TTF
            b"OTTO",  # OTF
            b"true",  # TrueType
            b"typ1",  # Type1
        ]
        if magic not in valid_signatures:
            log.warning(f"Downloaded file may not be a valid font (magic bytes: {magic.hex()})")

        # Save to cache
        cache_path.write_bytes(font_data)
        log.info(f"Downloaded and cached font from URL: {cache_path}")
        return str(cache_path)

    except HTTPError as e:
        raise ConnectionError(f"Failed to download font from {url}: HTTP {e.code}") from e
    except URLError as e:
        raise ConnectionError(f"Failed to connect to download font: {e.reason}") from e


def get_web_font_path(
    font_name: str,
    source: str = "google_fonts",
    url: str = "",
    weight: str = "regular",
) -> str:
    """Get the path to a web font, downloading it if necessary.

    Args:
        font_name: Name of the font
        source: Source type ("google_fonts" or "url")
        url: URL for URL-based fonts
        weight: Font weight for Google Fonts

    Returns:
        Path to the font file

    Raises:
        ValueError: If source is invalid or font cannot be found
    """
    if source == "google_fonts":
        return download_google_font(font_name, weight)
    elif source == "url":
        if not url:
            raise ValueError("URL is required when source is 'url'")
        return download_font_from_url(url)
    else:
        raise ValueError(f"Invalid font source: {source}. Must be 'google_fonts' or 'url'")


def list_cached_fonts() -> list[str]:
    """List all fonts currently in the cache.

    Returns:
        List of cached font filenames
    """
    cache_dir = get_font_cache_dir()
    return [f.name for f in cache_dir.glob("*.ttf")] + [f.name for f in cache_dir.glob("*.otf")]


def clear_font_cache() -> int:
    """Clear all cached fonts.

    Returns:
        Number of files deleted
    """
    cache_dir = get_font_cache_dir()
    count = 0
    for f in cache_dir.glob("*"):
        if f.is_file():
            f.unlink()
            count += 1
    log.info(f"Cleared font cache: {count} files deleted")
    return count
