# Agent CLI Command

The `nodetool agent` command allows you to run AI agents from the command line using a planning approach. The agent breaks down your objective into steps and executes them using the TaskPlanner.

## Quick Start

### Basic Usage

Run an agent with a simple prompt:

```bash
nodetool agent "Research the latest AI developments and create a summary"
```

### Using Configuration Files

Create a YAML configuration file to customize your agent:

```yaml
# my_agent.yaml
name: "Research Assistant"
provider: "openai"
model: "gpt-4o"
tools:
  - "google_search"
  - "browser"
max_steps: 15
verbose: true
```

Run with configuration:

```bash
nodetool agent "Research AI trends" --config my_agent.yaml
```

### Command-Line Overrides

Override configuration values from the command line:

```bash
nodetool agent "Analyze data" \
  --config agent.yaml \
  --model gpt-4o-mini \
  --max-steps 10 \
  --verbose
```

## Configuration Options

### YAML Configuration File

A complete agent configuration file supports the following fields:

```yaml
name: "Agent Name"                    # Required: Agent name
description: "Agent description"      # Optional: Description

system_prompt: |                      # Optional: Custom system prompt
  You are a helpful assistant...

provider: "openai"                    # LLM provider (default: "openai")
                                      # Options: openai, anthropic, ollama, 
                                      #          huggingface, google

model: "gpt-4o"                       # Model name (default: "gpt-4o")

planning_model: "gpt-4o"              # Optional: Model for planning
reasoning_model: "gpt-4o"             # Optional: Model for reasoning

tools:                                # Optional: List of tool names
  - "google_search"
  - "browser"
  - "calculator"
  - "python"

max_steps: 10                         # Maximum steps (default: 10)
max_step_iterations: 5                # Max iterations per step (default: 5)
max_token_limit: 100000               # Optional: Token limit before summarization

verbose: true                         # Enable verbose output (default: true)

docker_image: "nodetool"              # Optional: Docker image for isolation
use_sandbox: false                    # Optional: Use macOS sandbox (default: false)
```

## Available Tools

The following tools can be specified in the `tools` list:

### Search & Web
- `google_search` - Google search capability
- `google_news` - Google News search
- `google_images` - Google Images search
- `search` - Alias for google_search

### Browser Automation
- `browser` - Navigate and extract content from web pages
- `browser_navigate` - Navigate to URLs
- `browser_screenshot` - Take screenshots
- `browser_query` - Query page elements
- `browser_click` - Click elements
- `browser_type` - Type into form fields

### Filesystem
- `filesystem` - Read files (alias)
- `read_file` - Read file contents
- `write_file` - Write to files
- `list_directory` - List directory contents
- `delete_file` - Delete files

### Code Execution
- `python` - Execute Python code
- `execute_python` - Execute Python code (alias)

### Math & Calculations
- `calculator` - Perform calculations
- `math` - Perform calculations (alias)

### Vector Database (ChromaDB)
- `chroma_index` - Index documents in ChromaDB
- `chroma_search` - Search ChromaDB collections
- `chroma_markdown` - Split and index markdown

## Examples

### Research Agent

```yaml
# research_agent.yaml
name: "Research Assistant"
description: "Agent for comprehensive research tasks"

system_prompt: |
  You are a research assistant. Focus on:
  1. Finding accurate, reliable information
  2. Citing sources
  3. Providing comprehensive analysis

provider: "openai"
model: "gpt-4o"

tools:
  - "google_search"
  - "browser"
  - "write_file"

max_steps: 20
verbose: true
```

Usage:
```bash
nodetool agent "Research quantum computing advances in 2024" \
  --config research_agent.yaml
```

### Code Helper Agent

```yaml
# code_helper.yaml
name: "Code Helper"
provider: "anthropic"
model: "claude-3-5-sonnet-20241022"

tools:
  - "python"
  - "read_file"
  - "write_file"
  - "calculator"

max_steps: 15
```

Usage:
```bash
nodetool agent "Analyze this Python script and suggest improvements" \
  --config code_helper.yaml
```

### Simple Calculator Agent

```yaml
# simple_agent.yaml
name: "Calculator"
provider: "openai"
model: "gpt-4o-mini"

tools:
  - "calculator"
  - "python"

max_steps: 5
```

Usage:
```bash
nodetool agent "Calculate the compound interest on $10,000 at 5% for 10 years" \
  --config simple_agent.yaml
```

## Command Options

```
nodetool agent [OPTIONS] PROMPT

Arguments:
  PROMPT              The objective/prompt for the agent [required]

Options:
  -c, --config PATH   Path to agent configuration YAML file
  --provider TEXT     LLM provider (overrides config)
  --model TEXT        Model to use (overrides config)
  -v, --verbose       Enable verbose output (overrides config)
  --max-steps INT     Maximum number of steps (overrides config)
  --tools TEXT        Comma-separated list of tools (overrides config)
  --help              Show this message and exit
```

## Environment Variables

Make sure to set appropriate API keys for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY="your-key-here"

# Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# Google
export GEMINI_API_KEY="your-key-here"

# Ollama (usually local)
export OLLAMA_API_URL="http://localhost:11434"
```

## Tips

1. **Start Simple**: Begin with minimal tools and increase as needed
2. **Use Verbose Mode**: Enable `verbose: true` to see detailed execution steps
3. **Tool Selection**: Only include tools that are relevant to your task
4. **Model Selection**: Use faster/cheaper models for simple tasks, more capable models for complex reasoning
5. **Iteration Limits**: Increase `max_steps` for complex, multi-step tasks

## Troubleshooting

### "Unknown tool" Error
Make sure the tool name is in the list of available tools. Run `nodetool agent --help` to see documentation.

### API Key Errors
Verify your API keys are set correctly:
```bash
nodetool info  # Shows which API keys are configured
```

### Agent Not Making Progress
- Increase `max_steps` if the agent runs out of steps
- Try a more capable model
- Simplify your prompt or break it into smaller tasks

## See Also

- [Example configurations](./examples/)
- [Agent documentation](https://docs.nodetool.ai)
- [Tool documentation](https://docs.nodetool.ai/tools)
