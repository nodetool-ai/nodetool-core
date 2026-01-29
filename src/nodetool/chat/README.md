[← Back to Docs Index](../../../docs/index.md)

# Nodetool Chat Module

This directory contains the functionality for Nodetool's interactive chat interface, enabling conversational interaction
with AI models and integrated tools.

## Overview

The chat module provides:

- An interactive Command Line Interface (CLI) for chatting with different AI models.
- Support for multiple AI providers (e.g., OpenAI, Anthropic, Gemini, Ollama) through a provider abstraction layer.
- Integration with various tools (e.g., Web Search, Browser Interaction, File System Operations) that the AI can use
  during the conversation.
- User-defined workspace support for safe file system operations initiated by the chat agent.
- Management of chat history, model selection, and other session settings.

## Structure

- **`chat.py`**: Contains core chat logic, including handling multi-provider interactions, tool execution (`run_tool`,
  `run_tools`), and message serialization.
- **`chat_cli.py`**: Implements the interactive command-line interface using `prompt_toolkit` and `rich`. Defines
  commands (`/help`, `/model`, `/agent`, `/debug`, etc.) and handles user input and agent responses.
- **`regular_chat.py`**: Implements the logic for standard, non-agentic chat interactions.
- **`providers/`**: Contains modules for specific AI providers (e.g., `openai.py`, `anthropic.py`, `ollama.py`),
  implementing the `ChatProvider` interface for each.
- **`ollama_service.py`**: Provides specific functions for interacting with a local Ollama service (e.g., listing
  available Ollama models).
- **`dataframes.py`**: Includes utility functions for generating JSON schemas related to dataframe structures, likely
  used for tool interactions involving tabular data.
- **`help.py`**: Likely contains detailed help text or command documentation used by the CLI or other parts of the chat
  system.

## Key Features

- **Multi-Provider Support**: Easily switch between different LLM providers (OpenAI, Anthropic, Ollama, Gemini).
- **Tool Integration**: Allows the AI agent to leverage tools like web search, browsing, file downloads, PDF processing,
  and workspace commands.
- **Interactive CLI**: User-friendly command-line interface with history, auto-completion, and rich text formatting.
- **User-Defined Workspaces**: File operations require a user-configured workspace. If no workspace is assigned,
  disk I/O operations will throw a `PermissionError`.
- **Agentic Capabilities**: Supports agentic workflows where the AI can plan and execute multiple steps involving tools.
