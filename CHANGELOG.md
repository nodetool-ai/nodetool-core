# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2025-04-07

### Added

- Package uninstall request handling
- Rich library for enhanced terminal output
- Networkx dependency for improved functionality
- Data lineage in sub-tasks
- Playwright tool and debug mode
- Detailed documentation for message structures and tool call flows
- Torch availability check in BaseNode property handling
- Documentation generation CLI with module name option
- Node search and package discovery endpoints to Registry
- Production mode to CLI
- Repository ID extraction in package scanning
- Tomli library for TOML file parsing
- Protobuf dependency
- Support for Optional type handling in type_to_string conversion
- Project initialization command to CLI
- Namespace support for package installation
- Local package installation support
- Git commit hash tracking for package metadata
- Tabulate dependency for consistent output in package listing

### Changed

- Refactored imports in various modules
- Updated example workflows and enhanced descriptions
- Improved error handling in multiple modules
- Enhanced task update handling and refactored workflows
- Updated dependencies in pyproject.toml
- Refactored ChatCLI and enhanced agent execution flow
- Enhanced agent functionality and updated README for clarity
- Refactored email processing and CLI functionality
- Enhanced task management and file handling in chat module
- Updated Anthropic model version
- Refactored task management and enhanced planning capabilities
- Implemented workspace management in chat CLI
- Enhanced chat CLI and provider functionality
- Refactored chat module and CoT agent functionality
- Improved property assignment error handling in BaseNode
- Refactored environment configuration in CLI and environment module
- Optimized torch import handling
- Renamed CLI script from 'node-pack' to 'nodetool'
- Simplified Jekyll configuration for documentation generation
- Refactored documentation generation to use Jekyll for GitHub Pages
- Updated Python version constraint in project initialization
- Refactored node package scanning and discovery
- Improved type handling in code generation module
- Changed package registry to use JSON instead of YAML
- Refactored package registry and metadata handling
- Updated GitHub Actions workflow

### Removed

- Outdated project showcase and example code from README.md
- Prompt_engineering.md file
- Example scripts for improved clarity
- CoT agent implementation
- Unused JSON schema functions
- Redundant node directory creation in project initialization
- Package management CLI commands from cli.py
- Unused PDF indexing example script and test file
- Unused search and logical operator types
- Provider-specific nodes and dependencies
- macOS-specific nodes and dependencies
- export_requirements.sh and requirements.txt

### Fixed

- Ollama agent functionality
- Documentation generation issues
- Various test failures

### Security

- Changed project license from MIT to AGPL
