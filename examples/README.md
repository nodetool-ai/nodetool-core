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

### 1. Node Execution Examples

#### Simple Node Execution (`simple_node_execution.py`)

- Demonstrates direct execution of individual nodes
- Shows math and text processing operations
- Best starting point for understanding the node system

#### Node Tool Example (`node_tool_example.py`)

- Shows how to wrap workflow nodes as agent tools
- Demonstrates custom node creation

### 2. Agent Examples

#### Simple Agent (`test_simple_agent.py`)

- Basic agent setup with browser tool
- Demonstrates agent execution flow

#### Google Agent (`test_google_agent.py`)

- Google search capabilities
- Web research patterns

#### Google Grounded Agent (`test_google_grounded_agent.py`)

- Grounded search with source verification
- Research with citations

#### Google News Agent (`test_google_news_agent.py`)

- News search and aggregation
- Current events research

#### OpenAI Web Search (`test_openai_web_search.py`)

- OpenAI integration with web search
- Combined AI and search functionality

### 3. Workflow & Planning Examples

#### Graph Planner Simple Tests (`graph_planner_simple_tests.py`)

- Basic workflow planning patterns
- Simple agent workflows

#### Graph Planner Integration (`graph_planner_integration.py`)

- Complex workflow integration
- Multi-step planning

#### Graph Planner Preview Pattern (`graph_planner_preview_pattern.py`)

- Workflow preview capabilities
- Planning visualization

#### Graph Planner RAG Pattern (`graph_planner_rag_pattern.py`)

- RAG (Retrieval-Augmented Generation) workflows
- Knowledge base integration

#### Graph Planner Web Research (`graph_planner_web_research.py`)

- Web research workflows
- Automated information gathering

#### Graph Planner Data Transform (`graph_planner_data_transform.py`)

- Data transformation workflows
- ETL patterns

### 4. AI Provider Examples

#### OpenRouter Example (`openrouter_example.py`)

- Multi-provider routing via OpenRouter
- Model selection patterns

#### Minimax Example (`minimax_example.py`)

- Minimax provider integration
- Cost optimization patterns

#### HuggingFace OAuth (`huggingface_oauth_example.md`)

- HuggingFace authentication
- OAuth setup guide

### 5. Research & Analysis Agents

#### ChromaDB Research Agent (`chromadb_research_agent.py`)

- Document indexing with ChromaDB
- Semantic search capabilities

#### Wikipedia Agent (`wikipedia_agent_example.py`)

- Wikipedia research automation
- Structured documentation generation

#### Reddit Scraper Agent (`reddit_scraper_agent.py`)

- Reddit data collection
- Community analysis

#### Product Hunt Extractor (`product_hunt_extractor.py`)

- Product research automation
- Trend analysis

#### AI Workflows on Reddit (`ai_workflows_on_reddit.py`)

- Social media AI workflows
- Community engagement patterns

### 6. Task Planning Examples

#### Task Planner Batch (`test_task_planner_batch.py`)

- Batch task planning
- Multi-agent coordination

#### Learning Path Generator (`learning_path_generator.py`)

- Educational content planning
- Curriculum design

#### Logic Puzzle Solver (`solve_logic_puzzle.py`)

- Logical reasoning workflows
- Problem-solving patterns

### 7. Other Examples

#### Email Agent (`test_email_agent.py`)

- Email processing capabilities
- Communication automation

#### HackerNews Agent (`test_hackernews_agent.py`)

- Tech community analysis
- Content aggregation

#### Workspace Agent (`test_workspace_agent.py`)

- Workspace management
- File operations

## Running the Examples

To run any of these examples:

1. Ensure you have NodeTool installed and configured
1. Set up the required API keys for the chosen providers
1. Run the script using Python:

```bash
python examples/simple_node_execution.py
```

Each example will:

- Create a workspace directory for storing outputs (if needed)
- Connect to the specified AI provider (if applicable)
- Execute the tasks and display progress
- Save results to the workspace directory (if applicable)

## Key Concepts

- **Nodes**: Individual processing units that perform specific operations
- **Context**: ProcessingContext manages workspace, user, and execution state
- **Providers**: AI provider integrations (OpenAI, HuggingFace, etc.)
- **Agents**: AI assistants configured with specific tools and objectives

## Next Steps

After exploring these examples, you can:

1. Create custom nodes for your specific use cases
2. Build complex workflows using the graph planning system
3. Develop new agents with specialized tools
4. Integrate these capabilities into your applications

For more information, refer to the main NodeTool documentation.
