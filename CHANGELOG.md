# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2025-12-20

### Added

- Streaming node execution for real-time workflow processing
- Agent message metadata enrichment
- Comprehensive help documentation for built-in nodes
- HuggingFace search helper utilities
- Token counting utilities for LLM optimization
- Model query tool for agent workflows
- Model pack support for bundled model downloads
- Video frame extraction utility
- Command alias for package in CLI
- Enhanced Hugging Face integration logging
- Encrypted secret management with caching and invalidation
- Multiple WebSocket client support for downloads
- Documentation consolidation at docs.nodetool.ai

### Changed

- **Media Utilities Consolidation**: Refactored media/asset conversion utilities out of `processing_context.py` into
  focused, reusable modules:
  - Added `common/font_utils.py` with `get_system_font_path()` for cross-platform font discovery
  - Extended `common/image_utils.py` with `pil_image_to_base64_jpeg()` and `image_data_to_base64_jpeg()` for consistent
    image processing
  - Extended `common/video_utils.py` with `export_to_video_bytes()` for in-memory video encoding
  - Updated `processing_context.py` to delegate to shared utilities for `numpy_to_pil_image`, `get_system_font_path`,
    and video encoding
  - Consolidated duplicate `numpy_to_audio_segment` implementations across packages to use `common/audio_helpers`
  - Updated `ollama_provider.py` to use shared image utilities for JPEG conversion and resizing
- Standardized image handling to base64 for all providers
- Enhanced logging system with integrated display manager
- Improved execution tracking with structured events
- Refactored agent system for Step-based execution (renamed SubTask to Step)
- Simplified TaskPlanner to single-phase planning
- Updated all tests and examples for Step-based agent system
- Improved provider structured output support
- Enhanced ChromaDB collection handling and tenant management
- Improved help system with workflow patterns and token-efficient search
- Enhanced node search with phrase matching and better scoring
- Removed function calls from content in tool call parsing
- Updated Anthropic SDK to >=0.75.0
- Refactored Hugging Face download manager for improved robustness
- Changed default port to 7777
- Streamlined HuggingFace model discovery
- Refactored ModelManager for improved iteration and clarity
- Enhanced SQLiteConnectionPool to support multiple event loops
- Refined database operation logging
- Improved execution infrastructure and port configuration
- Restructured documentation and consolidated concepts
- Enabled unstructured subtasks with new default system prompt

### Fixed

- Safe MLX import with consistent error handling
- Robust offline cache and patchable downloads for HuggingFace
- ModelManager.set_model legacy signature restoration
- Normalized decrypt failures in security module
- Stabilized agent execution and tool schemas
- Acceptance of legacy instructions payloads in models
- Property schema and assignment backward compatibility in workflows
- Restored backward-compatible message/prediction fields in metadata
- Exported arg mapping and async run_graph in DSL
- ChromaDB collection handling issues
- Job execution problems
- Collection, file, and job API fixes
- Admin API and storage API issues
- Workspace checks
- HuggingFace cache tests for new DownloadManager API
- Port configuration issues
- Model manager iteration bugs
- Import-time crash with aiodns by pinning pycares version
- Ollama model retrieval with specific connection error messages

### Deprecated

- Direct imports of `numpy_to_audio_segment` from `nodetool-lib-data/src/nodetool/nodes/lib/numpy/utils.py` - use
  `from nodetool.media.audio.audio_helpers import numpy_to_audio_segment` instead

### Removed

- StepExecutor deprecated SubTaskContext implementation
- Deprecated example files
- CoT agent implementation (replaced with Step-based system)

## [November 2025]

### Added

- Model cache for disk-based caching of model metadata
- CLI command to list all cached HuggingFace models
- New HuggingFace model patterns and matchers
- Support for Flux and Qwen image types
- User-specific Supabase configuration options

### Changed

- Hosted Docker image usage
- HuggingFace model search functionality refactored
- ModelManager iteration improvements
- Improved model loading and validation

### Fixed

- Unreferenced variable issues
- Model loading edge cases

## [October 2025]

### Added

- New graph planner pattern examples
- Debugging reproduction scripts

### Changed

- Example workflows updated for new patterns
- Documentation improvements

### Removed

- Obsolete example scripts

## [September 2025]

### Added

- Background job system with async handling and backpressure
- MLX provider with audio and vision model support
- Allow missing properties option for node validation
- Custom serializer for default values in Property model
- Docker-based code runners for multiple languages
- NodeInstanceTool for dynamic output handling
- Memory URI cache with thread-safety improvements
- Image gen workflows API

### Changed

- Refactored to use TypedDict for output types
- Enhanced tool selection and UI proxy integration
- Improved async handling across API modules
- Migrated to async Chroma client
- Moved modules from common/ into domain packages
- Enhanced provider support and message handling
- Streamlined input/output checks
- Refactored agent tool processing

### Fixed

- Memory URI cache thread-safety for Agent subtasks
- Node loading error handling with enforcement
- Windows compatibility issues in multiple areas
- Async event loop on Windows
- Batch streaming nodes handling
- Numpy version compatibility

### Removed

- Pre-gathering method from BaseNode
- Unused packages from dependencies
- Browser tool from dependencies

## [May-August 2025]

### Added

- Help system consolidation with search guidelines
- Debug API endpoint for exporting bundles
- Cooperative shutdown features in stream runners
- Graceful WebSocket shutdown with Windows support
- Enhanced tool message processing
- OpenAI-compatible message formatting utilities
- Llama.cpp support with server manager and model fetching
- Verbose logging option for CLI commands
- HTTP default headers and timeout configuration
- Persistent event loop for WebSocket runner
- Image handling utilities with shared URI fetching
- ThreadedEventLoop for improved thread management
- Server subprocess runner (non-Docker)
- Actor-based workflow engine

### Changed

- Migrated from Poetry to Hatch for package management
- Transitioned to uv for dependency management
- Enhanced schema migration handling in SQLite adapter
- Improved workflow execution with pagination
- Refactored FastAPI lifecycle management
- Enhanced async file storage operations
- Migrated to async httpx for URI fetch
- Async database operations across all adapters
- Centralized dotenv loading and logging
- Improved provider model type handling
- Streamlined package node scanning
- Enhanced message content handling across providers

### Fixed

- Font file handling in ProcessingContext
- Asset query sanitization for underscores
- WorkflowRunner edge validation
- Global search for assets
- Agent tool error handling
- Windows file path handling in file API
- Uvicorn server reload on Windows
- ChromaDB collection handling
- Various async operation issues

### Removed

- Deprecated pipeline tags
- Claude auto review workflow
- Unused API routes
- Server subprocess runner (unused)
- Outdated RunPod deployment documentation

## [January 2025]

### Added

- Streaming chat response for help endpoint
- Production node security and metadata filtering
- JSON schema support for agents
- Float data type support in JSON schema generation
- Ollama embeddings support
- Collection indexing with parallel processing
- Semantic splitting for document processing

### Changed

- Refactored text handling to remove TextRef type
- Enhanced index endpoint with editor redirect
- Updated model recommendations across workflows
- Improved Electron integration

### Fixed

- Font path resolution in tests
- Collection creation with proper buffer and breakpoint settings
- Server health checks via HTTP on startup

## [October-December 2024]

### Added

- Initial core infrastructure development
- Foundation for workflow execution
- Basic provider integrations

### Changed

- Core architecture setup and refinements

### Fixed

- Early development stability improvements

## [February-April 2025]

### Added

- Core API infrastructure
- Database adapters (SQLite, PostgreSQL, Supabase)
- WebSocket support for real-time updates
- Initial provider integrations (OpenAI, Anthropic, Gemini, Ollama)
- Workflow execution engine
- Processing context system
- Asset and storage management
- Memory URI cache

### Changed

- Dependency management improvements
- Model manager import paths
- Initial architecture setup

### Fixed

- Initial bug fixes and stability improvements

## [September 2024]

### Added

- Initial agent framework infrastructure
- Task planning and execution system

### Changed

- Core architecture enhancements for agent support

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
