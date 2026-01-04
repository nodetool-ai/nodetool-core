# Debian Packaging for nodetool-core

This document describes how to build and publish nodetool-core as a Debian/Ubuntu package.

## Quick Start

### Building a .deb package locally

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y debhelper dh-python pybuild-plugin-pyproject \
    python3-all python3-setuptools python3-hatchling python3-pip

# Build the package (unsigned, for testing)
dpkg-buildpackage -us -uc -b

# The .deb file will be created in the parent directory
ls ../*.deb
```

### Installing the package locally

```bash
# Install the built package
sudo dpkg -i ../python3-nodetool-core_*.deb

# Install Python dependencies via pip (required)
pip3 install --user nodetool-core
```

## Directory Structure

```
debian/
├── changelog          # Package changelog (version history)
├── control            # Package metadata and dependencies
├── copyright          # License information
├── docs               # Additional documentation files
├── py3dist-overrides  # Python dependency mapping overrides
├── python3-nodetool-core.postinst  # Post-installation script
├── rules              # Build rules (Makefile)
├── scripts/           # Helper scripts for PPA publishing
└── source/
    ├── format         # Source package format (3.0 native)
    └── options        # Source package build options
```

## Phase 3: Publishing to Launchpad PPA

### Prerequisites

1. **Launchpad Account**: Create an account at https://launchpad.net
2. **GPG Key**: Generate and upload your GPG key to Launchpad
3. **PPA**: Create a Personal Package Archive

### GPG Key Setup

```bash
# Generate a new GPG key if you don't have one
gpg --full-generate-key

# List your keys
gpg --list-keys --keyid-format LONG

# Upload your public key to Ubuntu keyserver
gpg --keyserver keyserver.ubuntu.com --send-keys YOUR_KEY_ID

# Note: You need to add this key to your Launchpad account at:
# https://launchpad.net/~/+editpgpkeys
```

### Creating a PPA

1. Go to https://launchpad.net/~/+activate-ppa
2. Create a new PPA (e.g., "nodetool")
3. Note your PPA URL: `ppa:YOUR_USERNAME/nodetool`

### Building Source Package for PPA

```bash
# Navigate to the source directory
cd /path/to/nodetool-core

# Build a signed source package
debuild -S -sa

# This creates in the parent directory:
# - nodetool-core_0.6.2~rc18-1.dsc
# - nodetool-core_0.6.2~rc18-1.tar.xz (or .tar.gz)
# - nodetool-core_0.6.2~rc18-1_source.changes
```

### Uploading to PPA

```bash
# Upload to your PPA
dput ppa:YOUR_USERNAME/nodetool ../nodetool-core_0.6.2~rc18-1_source.changes

# Monitor build status at:
# https://launchpad.net/~YOUR_USERNAME/+archive/ubuntu/nodetool/+packages
```

### Multi-Version Ubuntu Support

To build for multiple Ubuntu versions, modify the changelog for each version:

```bash
# For Ubuntu 24.04 (Noble)
dch -D noble -v "0.6.2~rc18-1~noble1" "Release for Noble"
debuild -S -sa
dput ppa:YOUR_USERNAME/nodetool ../nodetool-core_0.6.2~rc18-1~noble1_source.changes

# For Ubuntu 22.04 (Jammy)
dch -D jammy -v "0.6.2~rc18-1~jammy1" "Release for Jammy"
debuild -S -sa
dput ppa:YOUR_USERNAME/nodetool ../nodetool-core_0.6.2~rc18-1~jammy1_source.changes

# For Ubuntu 20.04 (Focal)
dch -D focal -v "0.6.2~rc18-1~focal1" "Release for Focal"
debuild -S -sa
dput ppa:YOUR_USERNAME/nodetool ../nodetool-core_0.6.2~rc18-1~focal1_source.changes
```

### User Installation Instructions

Once published, users can install the package with:

```bash
# Add the PPA
sudo add-apt-repository ppa:YOUR_USERNAME/nodetool
sudo apt update

# Install the package
sudo apt install python3-nodetool-core

# Install Python dependencies (required)
pip3 install --user nodetool-core
```

## Versioning

The package version follows Debian versioning conventions:
- `0.6.2~rc18-1`: Version 0.6.2-rc.18, Debian revision 1
- The `~` character is used for pre-release versions (rc, beta, alpha)
- The `-1` suffix indicates the Debian packaging revision

## Troubleshooting

### Build Errors

1. **Missing build dependencies**: Install with `sudo apt-get build-dep nodetool-core`
2. **Python version issues**: Ensure Python 3.10+ is available
3. **Hatchling issues**: Install with `pip3 install hatchling`

### Launchpad Build Failures

1. Check build logs at: https://launchpad.net/~YOUR_USERNAME/+archive/ubuntu/nodetool/+builds
2. Common issues:
   - Missing build dependencies: Add them to `debian/control` Build-Depends
   - Version conflicts: Ensure dependency versions are available in Ubuntu repositories
   - Architecture issues: Use `Architecture: all` for pure Python packages

### Lintian Warnings

Current known warnings:
- `initial-upload-closes-no-bugs`: Expected for new packages not in Debian
- `no-manual-page`: Man page for `nodetool` command not included (optional)

## Dependency Notes

This package has extensive Python dependencies. Many are not available in Ubuntu repositories
and must be installed via pip. The post-installation script informs users about this requirement.

Key dependencies installed via pip:
- anthropic, openai, google-genai (AI providers)
- chromadb (vector database)
- huggingface-hub (model management)
- llama-index-core (LLM framework)
- And many more (see pyproject.toml for full list)
