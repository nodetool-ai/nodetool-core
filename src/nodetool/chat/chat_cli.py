"""
CLI interface for the Chain of Thought (CoT) agent.

This module provides a command-line interface for interacting with the CoT agent.
It supports multiple LLM providers, model selection, and tool management.

Features:
- Interactive chat with command history and tab completion
- Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
- Model selection and provider management
- Chain of Thought reasoning with step-by-step output
- Tool usage and execution
- Debug mode to display tool calls and results
"""

import asyncio
import json
import os
import tempfile
import traceback
import readline
from typing import List, Dict, Any, Sequence, Optional

from nodetool.chat.cot_agent import CoTAgent
from nodetool.chat.providers import get_provider, Chunk
from nodetool.chat.tools import Tool
from nodetool.chat.chat import (
    run_tool,
)
from nodetool.chat.regular_chat import process_regular_chat
from nodetool.metadata.types import Provider, Message, ToolCall, FunctionModel
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.chat.ollama_service import get_ollama_models
from nodetool.chat.chat import get_openai_models


async def chat_cli():
    """Interactive command-line chat interface with multi-provider support and CoT agent.

    Provides a CLI interface for interacting with various AI models through different providers.
    Features include:

    - Support for OpenAI, Anthropic, and Ollama providers
    - Model selection and provider switching
    - Command history with readline integration
    - Tab completion for commands and models
    - Tool execution capabilities
    - Streaming responses with real-time display
    - Chain of Thought (CoT) agent mode for step-by-step reasoning
    - Debug mode to display all tool calls and results

    Commands:
        /provider [openai|anthropic|ollama]: Switch between AI providers
        /model [model_name]: Select a specific model
        /models: List available models for the current provider
        /clear: Clear the conversation history
        /agent [on|off]: Toggle CoT agent mode
        /debug [on|off]: Toggle debug mode to display tool calls and results
        /quit or /exit: Exit the chat interface
        /help: Display available commands

    The CLI maintains conversation context and supports tool execution requested by AI models.
    Command history is preserved between sessions in ~/.nodetool_history.
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    context = ProcessingContext(user_id="test", auth_token="test")
    ollama_models = await get_ollama_models()
    openai_models = await get_openai_models()

    # Define default models for each provider
    default_models = {
        Provider.OpenAI: "gpt-4o",
        Provider.Anthropic: "claude-3-7-sonnet-20250219",
        Provider.Ollama: "llama3.2:3b",
    }

    provider = Provider.OpenAI
    model = FunctionModel(name=default_models[provider], provider=provider)
    messages: list[Message] = []

    # CoT agent mode settings
    agent_mode = False
    cot_agent = None

    # Debug mode setting
    debug_mode = False

    # Set up readline
    histfile = os.path.join(os.path.expanduser("~"), ".nodetool_history")
    try:
        readline.read_history_file(histfile)
    except FileNotFoundError:
        pass

    # Enable tab completion
    COMMANDS = [
        "help",
        "quit",
        "exit",
        "provider",
        "model",
        "models",
        "clear",
        "agent",
        "debug",
    ]
    PROVIDERS = [p.value.lower() for p in Provider]
    AGENT_OPTIONS = ["on", "off"]
    DEBUG_OPTIONS = ["on", "off"]

    def completer(text, state):
        if text.startswith("/"):
            text = text[1:]
            options = [f"/{cmd}" for cmd in COMMANDS if cmd.startswith(text)]
            return options[state] if state < len(options) else None
        elif text.startswith("/model "):
            model_text = text.split()[-1]
            options = [
                f"/model {m}" for m in ollama_models if m.name.startswith(model_text)
            ]
            options.extend(
                [f"/model {m}" for m in openai_models if m.id.startswith(model_text)]
            )
            return options[state] if state < len(options) else None
        elif text.startswith("/provider "):
            provider_text = text.split()[-1]
            options = [
                f"/provider {p}" for p in PROVIDERS if p.startswith(provider_text)
            ]
            return options[state] if state < len(options) else None
        elif text.startswith("/agent "):
            agent_text = text.split()[-1]
            options = [
                f"/agent {opt}" for opt in AGENT_OPTIONS if opt.startswith(agent_text)
            ]
            return options[state] if state < len(options) else None
        elif text.startswith("/debug "):
            debug_text = text.split()[-1]
            options = [
                f"/debug {opt}" for opt in DEBUG_OPTIONS if opt.startswith(debug_text)
            ]
            return options[state] if state < len(options) else None
        return None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")

    # Initialize CoT agent with the default provider and model
    async def initialize_cot_agent():
        nonlocal cot_agent
        # create temp folder for workspace
        workspace_dir = tempfile.mkdtemp()

        provider_instance = get_provider(provider)
        cot_agent = CoTAgent(
            provider=provider_instance,
            model=model,
            workspace_dir=workspace_dir,
        )
        return cot_agent

    # Initialize the CoT agent
    cot_agent = await initialize_cot_agent()

    def find_tool_by_name(name: str) -> Optional[Tool]:
        if cot_agent:
            for tool in cot_agent.tools:
                if tool.name == name:
                    return tool
        return None

    # Process CoT agent responses
    async def process_cot_response(problem):
        """Process a problem with the CoT agent and display the step-by-step reasoning."""
        if not cot_agent:
            print("Error: CoT agent not initialized")
            return

        try:
            # Get and display the step-by-step reasoning
            async for item in cot_agent.solve_problem(problem, show_thinking=True):
                if isinstance(item, Chunk):
                    print(item.content, end="", flush=True)
                elif isinstance(item, ToolCall):
                    # Show tool usage
                    print(f"\n[{item.name}]: {json.dumps(item.args)}")

                    # Execute the tool
                    tool_result = await run_tool(context, item, cot_agent.tools)
                    if debug_mode:
                        print(f"Result: {json.dumps(tool_result.result, indent=2)}")

            print("\n")  # Add a final newline
        except Exception as e:
            print(f"\nError during CoT reasoning: {e}")
            print(traceback.format_exc())

    print("Chat CLI - Type /help for commands")
    print(f"Using {provider.value} with model {model.name}")
    print("Agent mode is OFF (use /agent on to enable Chain of Thought reasoning)")
    print("Debug mode is OFF (use /debug on to display tool calls and results)")

    while True:
        try:
            user_input = input("> ").strip()
            readline.write_history_file(histfile)

            if user_input.startswith("/"):
                cmd_parts = user_input[1:].lower().split()
                cmd = cmd_parts[0]
                args = cmd_parts[1:] if len(cmd_parts) > 1 else []

                if cmd == "quit" or cmd == "exit":
                    break
                elif cmd == "help":
                    print("Commands:")
                    print("  /provider [openai|anthropic|ollama] - Set the provider")
                    print("  /model [model_name] - Set the model")
                    print("  /models - List available models for the current provider")
                    print("  /agent [on|off] - Toggle Chain of Thought agent mode")
                    print(
                        "  /debug [on|off] - Toggle debug mode to display tool calls and results"
                    )
                    print("  /tools - List available tools")
                    print("  /tools [tool_name] - Show details about a specific tool")
                    print("  /clear - Clear chat history")
                    print("  /quit or /exit - Exit the chat")
                    continue
                elif cmd == "models":
                    try:
                        if provider == Provider.Ollama:
                            print("\nAvailable Ollama models:")
                            for model_info in ollama_models:
                                print(f"- {model_info.name}")
                        elif provider == Provider.OpenAI:
                            print("\nAvailable OpenAI models (partial list):")
                            for model_info in openai_models[:10]:  # Show only first 10
                                print(f"- {model_info.id}")
                        elif provider == Provider.Anthropic:
                            print("\nAvailable Anthropic models:")
                            print("- claude-3-opus-20240229")
                            print("- claude-3-sonnet-20240229")
                            print("- claude-3-haiku-20240307")
                            print("- claude-3-5-sonnet-2024102")
                            print("- claude-3-7-sonnet-20250219")
                        print()
                    except Exception as e:
                        print(f"Error listing models: {e}")
                    continue
                elif cmd == "provider":
                    if not args:
                        print(f"Current provider: {provider.value}")
                        print(f"Available providers: {[p.value for p in Provider]}")
                        continue

                    provider_name = args[0].capitalize()
                    try:
                        provider = Provider[provider_name]
                        # Update model to default for new provider
                        model = FunctionModel(
                            name=default_models[provider], provider=provider
                        )
                        # Reinitialize CoT agent with new provider
                        cot_agent = await initialize_cot_agent()
                        print(
                            f"Provider set to {provider.value} with default model {default_models[provider]}"
                        )
                    except KeyError:
                        print(
                            f"Invalid provider. Choose from: {[p.value for p in Provider]}"
                        )
                    continue
                elif cmd == "model":
                    if not args:
                        print(f"Current model: {model.name}")
                        continue

                    model_name = args[0]
                    model = FunctionModel(name=model_name, provider=provider)
                    # Reinitialize CoT agent with new model
                    cot_agent = await initialize_cot_agent()
                    print(f"Model set to {model_name}")
                    continue
                elif cmd == "clear":
                    messages = []
                    if cot_agent:
                        cot_agent.clear_history()
                    print("Chat history cleared")
                    continue
                elif cmd == "agent":
                    if not args:
                        print(
                            f"Agent mode is currently: {'ON' if agent_mode else 'OFF'}"
                        )
                        continue

                    if args[0].lower() == "on":
                        agent_mode = True
                        print("Agent mode turned ON - Using Chain of Thought reasoning")
                    elif args[0].lower() == "off":
                        agent_mode = False
                        print("Agent mode turned OFF - Using standard chat")
                    else:
                        print("Usage: /agent [on|off]")
                    continue
                elif cmd == "debug":
                    if not args:
                        print(
                            f"Debug mode is currently: {'ON' if debug_mode else 'OFF'}"
                        )
                        continue

                    if args[0].lower() == "on":
                        debug_mode = True
                        print(
                            "Debug mode turned ON - Will display tool calls and results"
                        )
                    elif args[0].lower() == "off":
                        debug_mode = False
                        print("Debug mode turned OFF - Tool calls and results hidden")
                    else:
                        print("Usage: /debug [on|off]")
                    continue
                else:
                    print("Unknown command. Type /help for available commands")
                    continue

            # Skip empty input
            if not user_input:
                continue

            # Process with CoT agent if agent mode is enabled
            if agent_mode:
                await process_cot_response(user_input)  # type: ignore
                continue

            # Regular chat flow - Use the refactored module
            messages = await process_regular_chat(
                user_input=user_input,
                messages=messages,
                model=model,
                context=context,
                debug_mode=debug_mode,
            )

        except KeyboardInterrupt:
            return
        except EOFError:
            return
        except Exception as e:
            print(f"Error: {e}")
            stacktrace = traceback.format_exc()
            print(f"Stacktrace: {stacktrace}")


if __name__ == "__main__":
    asyncio.run(chat_cli())
