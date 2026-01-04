#!/bin/bash
#
# Build and upload nodetool-core to a Launchpad PPA for multiple Ubuntu versions
#
# Usage: ./build-ppa.sh [--upload] [--versions "noble jammy focal"]
#
# Prerequisites:
#   - GPG key configured for signing
#   - dput configured with your PPA
#   - debuild, dch installed
#

set -e

# Configuration
PACKAGE_NAME="nodetool-core"

# Extract version from pyproject.toml and convert to Debian format
# e.g., "0.6.2-rc.18" -> "0.6.2~rc18"
if command -v python3 &> /dev/null && [ -f "pyproject.toml" ]; then
    RAW_VERSION=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
print(data['project']['version'])
" 2>/dev/null || echo "")
    if [ -n "$RAW_VERSION" ]; then
        # Convert PEP 440 to Debian version: replace -rc. with ~rc
        BASE_VERSION=$(echo "$RAW_VERSION" | sed 's/-rc\./~rc/g' | sed 's/-/~/g')
    else
        BASE_VERSION="0.6.2~rc18"  # fallback
    fi
else
    BASE_VERSION="0.6.2~rc18"  # fallback
fi

DEBIAN_REVISION="1"

# PPA configuration - set PPA_NAME environment variable or use --ppa argument
# Example: export PPA_NAME="ppa:yourusername/nodetool"
PPA_NAME="${PPA_NAME:-}"

# Ubuntu versions to build for (default: latest LTS versions)
DEFAULT_VERSIONS="noble jammy"

# Parse arguments
UPLOAD=false
VERSIONS="$DEFAULT_VERSIONS"

while [[ $# -gt 0 ]]; do
    case $1 in
        --upload)
            UPLOAD=true
            shift
            ;;
        --versions)
            VERSIONS="$2"
            shift 2
            ;;
        --ppa)
            PPA_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--upload] [--versions 'noble jammy'] [--ppa ppa:user/name]"
            echo ""
            echo "Options:"
            echo "  --upload    Upload to PPA after building"
            echo "  --versions  Space-separated list of Ubuntu versions (default: '$DEFAULT_VERSIONS')"
            echo "  --ppa       PPA to upload to (default: '$PPA_NAME')"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Validate PPA configuration for uploads
if [ "$UPLOAD" = true ] && [ -z "$PPA_NAME" ]; then
    echo "ERROR: PPA name is required for uploads."
    echo ""
    echo "Set the PPA name using one of these methods:"
    echo "  1. Environment variable: export PPA_NAME='ppa:yourusername/nodetool'"
    echo "  2. Command line argument: $0 --upload --ppa 'ppa:yourusername/nodetool'"
    echo ""
    echo "To create a PPA, visit: https://launchpad.net/~/+activate-ppa"
    exit 1
fi

echo "============================================"
echo "Building $PACKAGE_NAME for PPA upload"
echo "============================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "Base version: $BASE_VERSION-$DEBIAN_REVISION"
echo "Ubuntu versions: $VERSIONS"
echo "PPA: ${PPA_NAME:-'(not set - upload disabled)'}"
echo "Upload: $UPLOAD"
echo ""

# Create temporary build directory
BUILD_DIR="/tmp/${PACKAGE_NAME}-ppa-build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Copy source to build directory
echo "Copying source to build directory..."
cp -r "$PROJECT_ROOT" "$BUILD_DIR/$PACKAGE_NAME"
cd "$BUILD_DIR/$PACKAGE_NAME"

# Clean any existing build artifacts
rm -rf .pybuild/ debian/python3-nodetool-core/ debian/.debhelper/
rm -f debian/debhelper-build-stamp debian/*.substvars debian/files

# Store original changelog
cp debian/changelog debian/changelog.orig

for VERSION in $VERSIONS; do
    echo ""
    echo "============================================"
    echo "Building for Ubuntu $VERSION"
    echo "============================================"
    
    # Restore original changelog
    cp debian/changelog.orig debian/changelog
    
    # Update version for this Ubuntu release
    FULL_VERSION="${BASE_VERSION}-${DEBIAN_REVISION}~${VERSION}1"
    
    # Update changelog
    dch -D "$VERSION" -v "$FULL_VERSION" "Build for Ubuntu $VERSION"
    
    echo "Building source package version $FULL_VERSION..."
    
    # Clean and build source package
    debian/rules clean 2>/dev/null || true
    debuild -S -sa --no-check-builddeps
    
    if [ "$UPLOAD" = true ]; then
        echo "Uploading to $PPA_NAME..."
        CHANGES_FILE="../${PACKAGE_NAME}_${FULL_VERSION}_source.changes"
        if [ -f "$CHANGES_FILE" ]; then
            dput "$PPA_NAME" "$CHANGES_FILE"
            echo "Upload complete for $VERSION!"
        else
            echo "ERROR: Changes file not found: $CHANGES_FILE"
            ls -la ../*.changes 2>/dev/null || echo "No changes files found"
        fi
    else
        echo "Source package built. Use --upload to upload to PPA."
    fi
done

# Restore original changelog
cp debian/changelog.orig debian/changelog
rm debian/changelog.orig

echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""
echo "Built packages in: $BUILD_DIR"
ls -la "$BUILD_DIR"/*.changes 2>/dev/null || echo "No changes files found"
echo ""

if [ "$UPLOAD" = false ]; then
    echo "To upload, run: $0 --upload --ppa $PPA_NAME"
fi

echo ""
echo "Monitor build status at:"
echo "  https://launchpad.net/${PPA_NAME#ppa:}/+packages"
