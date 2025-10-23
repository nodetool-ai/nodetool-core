[← Back to Docs Index](../docs/index.md)

# NodeTool Core Examples

This directory contains example scripts that demonstrate how to use NodeTool Core for various use cases.

## Prerequisites

Before running these examples, you'll need to:

1. Install NodeTool Core

   ```bash
   pip install nodetool-core
   ```

1. Set up necessary API keys

   ```
   nodetool settings edit
   ```

## Available Examples

### 1. Web Research and Analysis

#### Wikipedia-Style Research Agent (`wikipedia_agent_example.py`)

- Creates Wikipedia-style documentation through web research
- Structures content into well-organized markdown files
- Uses web search and browsing capabilities

#### ChromaDB Research Agent (`chromadb_research_agent.py`)

- Processes and indexes documents using ChromaDB
- Enables semantic search across indexed content
- Performs research using stored documents

#### HackerNews Agent (`test_hackernews_agent.py`)

- Analyzes HackerNews content and discussions
- Gathers insights from tech community discussions

### 2. Social Media Analysis Agents

#### Twitter/X Scraper (`twitter_scraper_agent.py`)

- Collects trending topics and viral content
- Analyzes social media patterns

#### Instagram Scraper (`instagram_scraper_agent.py`)

- Tracks trending content and hashtags
- Analyzes platform-specific trends

#### Reddit Scraper (`reddit_scraper_agent.py`)

- Gathers information from Reddit discussions
- Analyzes community trends and opinions

### 3. Professional Research Tools

#### LinkedIn Job Market Agent (`linkedin_job_market_agent.py`)

- Researches current job market trends
- Analyzes hiring patterns and industry demands

#### Google Search Agent (`test_google_agent.py`)

- Demonstrates advanced Google search capabilities
- Collects and organizes search results

### 4. Utility Agents

#### Email Processing (`test_email_agent.py`)

- Demonstrates email retrieval and processing
- Shows email content analysis capabilities

#### OpenAI Web Search (`test_openai_web_search.py`)

- Integrates OpenAI's capabilities with web search
- Shows combined AI and search functionality

## Running the Examples

To run any of these examples:

1. Ensure you have NodeTool installed and configured
1. Set up the required API keys for the chosen providers
1. Run the script using Python:

```bash
python examples/wikipedia_agent_example.py
```

Each example will:

- Create a workspace directory for storing outputs
- Connect to the specified AI provider
- Execute the tasks and display progress
- Save results to the workspace directory

## Key Concepts

- **Agent**: An AI assistant configured with specific tools and capabilities
- **Tools**: Components that enable agents to interact with external systems
- **Tasks & Subtasks**: Units of work defining what an agent needs to accomplish

## Next Steps

After exploring these examples, you can:

1. Create custom agents for your specific use cases
1. Develop new tools to expand agent capabilities
1. Build more complex multi-agent systems
1. Integrate these capabilities into your applications

For more information, refer to the main NodeTool documentation.
