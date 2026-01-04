# Ubuntu Package Publishing Implementation Guide for nodetool-core

## Project Context

**Project:** nodetool-core
**Type:** Python library for building and running AI workflows
**Repository:** /home/user/nodetool-core
**Current Version:** 0.6.2-rc.18
**Build System:** hatchling (pyproject.toml-based)
**Python Version:** >=3.10.13
**Git Branch:** claude/ubuntu-package-publishing-7x18B

### Project Structure
- Source code: `src/nodetool/`
- Tests: `tests/`
- Build config: `pyproject.toml`
- CLI entry point: `nodetool` command (defined in pyproject.toml scripts)

### Key Dependencies
The project has ~100+ Python dependencies including:
- Core: FastAPI, Pydantic, NetworkX, Docker
- AI: OpenAI, Anthropic, Hugging Face, Ollama
- Media: FFmpeg-python, Pillow, OpenCV
- Database: ChromaDB, PostgreSQL, SQLite
- System requirements: ffmpeg, pandoc (external binaries)

---

## Your Task: Implement Ubuntu Package Publishing (Phases 1-3)

Implement a complete Ubuntu packaging workflow that allows users to install nodetool-core as a system package using `apt` or `dpkg`.

---

## Phase 1: Quick .deb Package Creation

### Objective
Create a working .deb package quickly using automated tools for testing and validation.

### Requirements

1. **Choose a Tool:**
   - **Option A: fpm (Recommended)** - Simple, Ruby-based packager
   - **Option B: stdeb** - Python-specific Debian packager
   - **Option C: dh-python with pybuild** - Official Debian method

2. **Create Packaging Script:**
   - Create `scripts/build_deb.sh` that:
     - Builds the Python package (wheel/sdist)
     - Converts to .deb format
     - Handles dependencies appropriately
     - Outputs to `dist/` directory
   - Make the script idempotent and well-documented

3. **Handle System Dependencies:**
   - Map Python dependencies to Ubuntu packages where possible
   - Mark ffmpeg and pandoc as system dependencies
   - Document which dependencies are bundled vs. system-provided

4. **Package Metadata:**
   - Package name: `python3-nodetool-core` (follow Ubuntu naming convention)
   - Version: Extract from pyproject.toml
   - Description: Use project description
   - Maintainer: Use author info from pyproject.toml
   - Architecture: all (pure Python) or amd64 if needed
   - Section: python
   - Priority: optional

5. **Post-Install Validation:**
   - Ensure the `nodetool` CLI command is available in PATH
   - Verify Python package is importable: `python3 -c "import nodetool"`
   - Test basic functionality

### Deliverables for Phase 1
- [ ] `scripts/build_deb.sh` - Automated build script
- [ ] `scripts/test_deb.sh` - Script to test the built package
- [ ] `dist/python3-nodetool-core_*.deb` - Built package
- [ ] `docs/packaging/QUICK_DEB.md` - Documentation for this method

---

## Phase 2: Proper Debian Packaging

### Objective
Create standard Debian packaging files following Debian Python Policy for PPA compatibility and maintainability.

### Requirements

1. **Create debian/ Directory Structure:**
   ```
   debian/
   ├── changelog          # Version history in Debian format
   ├── control            # Package metadata and dependencies
   ├── rules              # Build instructions (Makefile)
   ├── compat             # Debhelper compatibility level (13 or 14)
   ├── copyright          # License information
   ├── source/
   │   └── format         # Source format (3.0 quilt or native)
   ├── python3-nodetool-core.install  # File installation rules
   └── python3-nodetool-core.manpages # Man pages if available
   ```

2. **debian/control Requirements:**
   - **Source package:**
     - Source: nodetool-core
     - Section: python
     - Priority: optional
     - Maintainer: Extract from pyproject.toml
     - Build-Depends: debhelper-compat (= 13), dh-python, python3-all, python3-setuptools, python3-hatchling
     - Standards-Version: 4.6.2
     - Homepage: GitHub URL if available
     - Vcs-Git: Repository URL
     - Vcs-Browser: Repository browser URL

   - **Binary package:**
     - Package: python3-nodetool-core
     - Architecture: all (or amd64 if native extensions exist)
     - Depends: ${python3:Depends}, ${misc:Depends}, ffmpeg, pandoc, python3-pip
     - Recommends: (optional dependencies)
     - Suggests: (enhanced features)
     - Description: Multi-line description (short + long)

3. **debian/rules Requirements:**
   ```makefile
   #!/usr/bin/make -f

   %:
   	dh $@ --with python3 --buildsystem=pybuild

   override_dh_auto_install:
   	dh_auto_install
   	# Any custom installation steps

   override_dh_auto_test:
   	# Skip tests during package build or run specific tests
   	@echo "Skipping tests during package build"
   ```
   - Use dh_python3 for automatic Python dependency detection
   - Handle test suite appropriately (may need to skip heavy tests)
   - Ensure CLI script is installed correctly

4. **debian/changelog Format:**
   ```
   nodetool-core (0.6.2~rc.18-1) UNRELEASED; urgency=medium

     * Initial release for Ubuntu packaging
     * Added Debian packaging files

    -- Matthias Georgi <matti.georgi@gmail.com>  Sat, 04 Jan 2026 00:00:00 +0000
   ```
   - Use `dch` command to maintain it
   - Follow Debian version conventions (~ for rc/beta)
   - Include Ubuntu target (focal, jammy, noble, etc.)

5. **debian/copyright Requirements:**
   - Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
   - Include all license information from LICENSE file
   - List all copyright holders
   - Document third-party code if any

6. **Build Testing:**
   - Build source package: `dpkg-buildpackage -S`
   - Build binary package: `dpkg-buildpackage -b`
   - Test with lintian: `lintian *.deb` (fix all errors, minimize warnings)
   - Test installation: `sudo dpkg -i python3-nodetool-core_*.deb`
   - Test removal: `sudo apt-get remove python3-nodetool-core`

### Deliverables for Phase 2
- [ ] Complete `debian/` directory with all required files
- [ ] Clean lintian output (no errors)
- [ ] Tested .deb package that installs and removes cleanly
- [ ] `docs/packaging/DEBIAN_PACKAGING.md` - Comprehensive packaging documentation

---

## Phase 3: PPA (Personal Package Archive) Setup

### Objective
Set up automated publishing to a Launchpad PPA for easy user installation.

### Requirements

1. **Launchpad Prerequisites Documentation:**
   Create `docs/packaging/LAUNCHPAD_SETUP.md` documenting:
   - How to create a Launchpad account
   - GPG key generation and upload to Launchpad
   - SSH key setup for Launchpad
   - Creating a PPA (instructions for maintainer)
   - Setting up dput configuration

2. **Multi-Ubuntu Version Support:**
   - Target Ubuntu versions:
     - 20.04 LTS (Focal Fossa)
     - 22.04 LTS (Jammy Jellyfish)
     - 24.04 LTS (Noble Numbat)
     - 24.10 (Oracular Oriole) - current
   - Create separate debian/changelog entries for each version
   - Handle version-specific dependencies if needed

3. **PPA Upload Script:**
   Create `scripts/upload_to_ppa.sh` that:
   - Takes PPA name as argument (e.g., `ppa:username/nodetool`)
   - Validates GPG key is set up
   - For each Ubuntu version:
     - Updates debian/changelog with correct target
     - Builds source package (dpkg-buildpackage -S)
     - Signs package with GPG
     - Uploads to PPA using dput
   - Provides clear progress output
   - Validates upload success

4. **Version Management:**
   - Script to sync version from pyproject.toml to debian/changelog
   - Handle pre-release versions (rc, beta, alpha) correctly
   - Ensure PPA versions don't conflict with official packages

5. **GitHub Actions Workflow (Optional but Recommended):**
   Create `.github/workflows/ppa-publish.yml`:
   - Trigger: On release tags (v*.*.*)
   - Steps:
     - Checkout code
     - Import GPG key from GitHub secrets
     - Build source packages for all Ubuntu versions
     - Upload to PPA
   - Document required GitHub secrets

6. **User Installation Documentation:**
   Create `docs/packaging/INSTALL_FROM_PPA.md`:
   ```bash
   # Add PPA
   sudo add-apt-repository ppa:username/nodetool
   sudo apt update

   # Install
   sudo apt install python3-nodetool-core

   # Verify
   nodetool --version
   ```

7. **Testing PPA Packages:**
   - Document how to test packages in a clean Docker container
   - Create `scripts/test_ppa_install.sh` for automated testing
   - Test on multiple Ubuntu versions using Docker

### Deliverables for Phase 3
- [ ] `scripts/upload_to_ppa.sh` - PPA upload automation
- [ ] `scripts/test_ppa_install.sh` - PPA installation testing
- [ ] `docs/packaging/LAUNCHPAD_SETUP.md` - Maintainer setup guide
- [ ] `docs/packaging/INSTALL_FROM_PPA.md` - User installation guide
- [ ] `.github/workflows/ppa-publish.yml` - CI/CD workflow (optional)
- [ ] Updated README.md with PPA installation instructions

---

## Important Debian Packaging Policies

### Python Packages in Debian/Ubuntu
1. **Naming:** Binary packages must be named `python3-<packagename>`
2. **Dependencies:** Use ${python3:Depends} for auto-detected deps
3. **Installation:** Install to `/usr/lib/python3/dist-packages/` (automatic with dh_python3)
4. **Scripts:** CLI scripts go to `/usr/bin/`
5. **Documentation:** Man pages to `/usr/share/man/`

### Version Numbering
- Debian version format: `<upstream_version>-<debian_revision>`
- Pre-release versions: Use `~` (e.g., `0.6.2~rc.18-1`)
- PPA versions: Can append `~ppa1`, `~ubuntu1`, etc.

### Dependency Handling
- **Build-Depends:** Tools needed to build the package
- **Depends:** Required runtime dependencies
- **Recommends:** Important but not mandatory
- **Suggests:** Optional enhancements

### Common Issues to Avoid
1. **Don't bundle dependencies** - Use system packages where available
2. **Don't include .git directories** - Use debian/source/options to exclude
3. **Test on clean systems** - Use Docker or VMs
4. **Sign all uploads** - PPA requires GPG signatures
5. **Watch file size** - Large packages may time out on Launchpad

---

## Testing Checklist

### Phase 1 Testing
- [ ] .deb package builds without errors
- [ ] Package installs with `sudo dpkg -i`
- [ ] CLI command `nodetool` is available
- [ ] Python import works: `python3 -c "import nodetool"`
- [ ] Package removes cleanly with `sudo apt-get remove`

### Phase 2 Testing
- [ ] `dpkg-buildpackage -S` succeeds (source package)
- [ ] `dpkg-buildpackage -b` succeeds (binary package)
- [ ] `lintian *.deb` shows no errors
- [ ] Package metadata is correct (`dpkg -I *.deb`)
- [ ] File installation is correct (`dpkg -c *.deb`)
- [ ] Install/remove cycle works
- [ ] Dependency resolution is correct

### Phase 3 Testing
- [ ] Source package builds for all target Ubuntu versions
- [ ] Uploads to PPA succeed
- [ ] Launchpad builds packages successfully
- [ ] Test installation from PPA on each Ubuntu version
- [ ] Package updates work correctly
- [ ] Documentation is clear and complete

---

## File Structure After Completion

```
nodetool-core/
├── debian/                          # Debian packaging files
│   ├── changelog
│   ├── control
│   ├── rules
│   ├── compat
│   ├── copyright
│   ├── source/
│   │   └── format
│   └── python3-nodetool-core.install
├── scripts/
│   ├── build_deb.sh                 # Phase 1: Quick .deb builder
│   ├── test_deb.sh                  # Phase 1: Test .deb package
│   ├── upload_to_ppa.sh             # Phase 3: Upload to PPA
│   └── test_ppa_install.sh          # Phase 3: Test PPA installation
├── docs/packaging/
│   ├── QUICK_DEB.md                 # Phase 1 documentation
│   ├── DEBIAN_PACKAGING.md          # Phase 2 documentation
│   ├── LAUNCHPAD_SETUP.md           # Phase 3 maintainer guide
│   └── INSTALL_FROM_PPA.md          # Phase 3 user guide
├── .github/workflows/
│   └── ppa-publish.yml              # Phase 3 CI/CD (optional)
└── README.md                        # Updated with installation instructions
```

---

## Execution Instructions

### Step-by-Step Implementation

1. **Start with Phase 1:**
   - Implement quick .deb building using fpm or stdeb
   - Test thoroughly on Ubuntu 24.04
   - Document the process

2. **Move to Phase 2:**
   - Create proper debian/ directory
   - Build and test packages manually
   - Fix all lintian issues
   - Document everything

3. **Complete with Phase 3:**
   - Set up PPA infrastructure
   - Create upload automation
   - Test end-to-end workflow
   - Write user documentation

### Git Workflow
- Work on branch: `claude/ubuntu-package-publishing-7x18B`
- Commit after each phase completion
- Push to origin when complete

### Success Criteria
The implementation is complete when:
1. A user can install nodetool-core with: `sudo add-apt-repository ppa:username/nodetool && sudo apt install python3-nodetool-core`
2. The `nodetool` CLI works correctly after installation
3. All documentation is complete and accurate
4. The packaging passes lintian checks
5. The workflow is reproducible and maintainable

---

## Additional Resources

### Debian Packaging Documentation
- Debian Python Policy: https://www.debian.org/doc/packaging-manuals/python-policy/
- Debian New Maintainer's Guide: https://www.debian.org/doc/manuals/maint-guide/
- Ubuntu Packaging Guide: https://packaging.ubuntu.com/html/

### Tools
- debhelper: https://manpages.debian.org/debhelper
- dh-python: https://wiki.debian.org/Python/Pybuild
- fpm: https://github.com/jordansissel/fpm
- lintian: https://lintian.debian.org/

### Launchpad
- Launchpad PPAs: https://help.launchpad.net/Packaging/PPA
- Uploading to PPA: https://help.launchpad.net/Packaging/PPA/Uploading

---

## Questions to Consider

1. **Dependency Strategy:** Should we bundle all Python dependencies or rely on Ubuntu packages where available?
2. **System Requirements:** How to handle ffmpeg and pandoc? Mark as dependencies or document separately?
3. **PPA Naming:** What PPA name to use? (e.g., ppa:nodetool/stable, ppa:nodetool/dev)
4. **Target Versions:** Should we support older Ubuntu versions (18.04, 16.04)?
5. **Testing:** Should we set up automated testing in Docker containers?

---

## Notes

- The project uses hatchling as build backend, which is well-supported by modern debhelper
- The package has many dependencies - consider creating a minimal variant if needed
- Version is currently 0.6.2-rc.18 (pre-release) - handle this appropriately in Debian versioning
- The CLI entry point is well-defined in pyproject.toml, which makes packaging easier
- Consider creating separate packages for optional dependencies (recommends/suggests)

---

## Expected Timeline

- **Phase 1:** 2-4 hours (quick implementation and testing)
- **Phase 2:** 4-8 hours (proper packaging, testing, documentation)
- **Phase 3:** 4-6 hours (PPA setup, multi-version support, automation)

**Total:** ~10-18 hours of focused work

---

## Deliverables Summary

By the end of this task, you should have:

1. ✅ Working .deb packages that can be installed locally
2. ✅ Proper Debian packaging following Ubuntu standards
3. ✅ PPA infrastructure for easy user installation
4. ✅ Comprehensive documentation for maintainers and users
5. ✅ Automated scripts for building and uploading
6. ✅ Testing infrastructure
7. ✅ Optional CI/CD integration

Good luck! Remember to test thoroughly at each phase before moving to the next.
