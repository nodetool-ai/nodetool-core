#!/bin/bash
#
# Build a local .deb package for testing
#
# Usage: ./build-local.sh
#

set -e

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================"
echo "Building nodetool-core .deb package"
echo "============================================"
echo ""

# Check for required tools
for tool in dpkg-buildpackage dh_python3; do
    if ! command -v "$tool" &> /dev/null; then
        echo "ERROR: $tool is not installed"
        echo "Install build dependencies with:"
        echo "  sudo apt-get install debhelper dh-python pybuild-plugin-pyproject python3-all python3-setuptools python3-hatchling python3-pip"
        exit 1
    fi
done

# Clean previous build artifacts
echo "Cleaning previous build artifacts..."
rm -rf .pybuild/ debian/python3-nodetool-core/ debian/.debhelper/
rm -f debian/debhelper-build-stamp debian/*.substvars debian/files

# Build the package (unsigned)
echo "Building package..."
dpkg-buildpackage -us -uc -b

# Find and display the built package
echo ""
echo "============================================"
echo "Build complete!"
echo "============================================"
echo ""

DEB_FILE=$(ls ../*.deb 2>/dev/null | head -1)
if [ -n "$DEB_FILE" ]; then
    echo "Package built: $DEB_FILE"
    echo ""
    echo "Package info:"
    dpkg -I "$DEB_FILE" | head -20
    echo ""
    echo "To install:"
    echo "  sudo dpkg -i $DEB_FILE"
    echo "  pip3 install --user nodetool-core  # Install Python dependencies"
else
    echo "ERROR: No .deb file found"
    exit 1
fi
