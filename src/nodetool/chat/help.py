"""
Nodetool Help System Module

This module implements the help and documentation system for Nodetool, providing:
- Documentation indexing and searching using ChromaDB for semantic search
- Example workflow indexing and retrieval
- Interactive chat-based help using LLMs
- Core documentation management

Key Components:
- Documentation indexing with semantic and keyword search capabilities
- Example workflow management and search
- Node property lookup and validation
- Interactive chat interface with tool-augmented responses

The module uses ChromaDB for vector storage and retrieval, and integrates with
Ollama for LLM-powered help responses. It supports both semantic and keyword-based
searches across documentation and examples.
"""

import asyncio
import json
import os
import re
from typing import Any, AsyncGenerator, Mapping, List, Type, Dict, cast
import ollama
import readline
import uuid
from pydantic import BaseModel

import chromadb
from chromadb.api.types import IncludeEnum, WhereDocument, Metadata
from jsonschema import validators
from nodetool.chat.providers.base import ChatProvider
from nodetool.common.settings import get_system_data_path
from nodetool.chat.ollama_service import get_ollama_client
from nodetool.common.environment import Environment
from nodetool.metadata.types import (
    Message,
    MessageTextContent,
    Provider,
    ToolCall,
)
from nodetool.packages.registry import Registry
from nodetool.workflows.types import Chunk
from nodetool.workflows.processing_context import ProcessingContext
from nodetool.agents.tools.base import Tool
from nodetool.chat.providers import get_provider


doc_folder = os.path.join(os.path.dirname(__file__), "docs")
examples = None
documentation = None

log = Environment.get_logger()


def validate_schema(schema):
    meta_schema = validators.Draft7Validator.META_SCHEMA

    # Create a validator
    validator = validators.Draft7Validator(meta_schema)

    try:
        # Validate the schema
        validator.validate(schema)
        print("The schema is valid.")
        return True
    except Exception as e:
        print(f"The schema is invalid. Error: {e}")
        return False


def get_collection(name) -> chromadb.Collection:
    """
    Get or create a collection with the given name.

    Args:
        context: The processing context.
        name: The name of the collection to get or create.
    """
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction  # type: ignore
    from chromadb.config import DEFAULT_DATABASE, DEFAULT_TENANT, Settings

    chroma_path = get_system_data_path("chroma-docs")

    log.info(f"Using collection {name} from {chroma_path}")

    client = chromadb.PersistentClient(
        path=Environment.get_chroma_path(),
        tenant=DEFAULT_TENANT,
        database=DEFAULT_DATABASE,
        settings=Settings(anonymized_telemetry=False),
    )

    embedding_function = SentenceTransformerEmbeddingFunction()

    return client.get_or_create_collection(
        name=name,
        embedding_function=embedding_function,  # type: ignore
        metadata={"embedding_model": "all-MiniLM-L6-v2"},
    )


def index_documentation(collection: chromadb.Collection):
    """
    Index the documentation if it doesn't exist yet.
    """

    registry = Registry()
    installed_packages = registry.list_installed_packages()

    ids = []
    docs = []
    metadata = []
    for package in installed_packages:
        if package.nodes:
            for node in package.nodes:
                ids.append(node.node_type)
                docs.append(node.description)
                node_meta_raw = node.model_dump()
                # Filter to include only valid Metadata types (str, int, float, bool)
                node_meta: Metadata = {
                    k: v
                    for k, v in node_meta_raw.items()
                    if isinstance(v, (str, int, float, bool))
                }
                metadata.append(node_meta)

    collection.add(ids=ids, documents=docs, metadatas=metadata)  # type: ignore
    return collection


def index_examples(collection: chromadb.Collection):
    """
    Index the examples if they don't exist yet.
    """
    from nodetool.workflows.examples import load_examples

    examples = load_examples()
    ids = [example.id for example in examples]
    docs = [example.model_dump_json() for example in examples]

    collection.add(ids=ids, documents=docs)
    print("Indexed examples")


def get_doc_collection():
    collection = get_collection("docs")
    if collection.count() == 0:
        index_documentation(collection)
    return collection


def get_example_collection():
    collection = get_collection("examples")
    if collection.count() == 0:
        index_examples(collection)

    return collection


class SearchResult(BaseModel):
    id: str
    content: str
    metadata: Mapping[str, Any] | None = None


def semantic_search_documentation(
    query: str,
) -> list[SearchResult]:
    """
    Perform semantic search on documentation using embeddings.

    Args:
        query: The query to search for.

    Returns:
        A list of search results from semantic matching.
    """
    n_results = 10
    collection = get_doc_collection()
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
    )

    search_results = []
    if (
        results["documents"]
        and results["ids"]
        and results["ids"][0]
        and len(results["documents"][0]) > 0
    ):
        for i, doc in enumerate(results["documents"][0]):
            current_metadata = None
            if (
                results["metadatas"]
                and results["metadatas"][0]
                and i < len(results["metadatas"][0])
            ):
                current_metadata = results["metadatas"][0][i]

            search_results.append(
                SearchResult(
                    id=results["ids"][0][i],
                    content=doc,
                    metadata=current_metadata,
                )
            )
    return search_results


def node_properties(node_type: str) -> Dict[str, Any]:
    """
    Get the properties of a node.
    """
    collection = get_doc_collection()
    results = collection.get(
        ids=[node_type], include=[IncludeEnum.documents, IncludeEnum.metadatas]
    )
    if (
        results["metadatas"]
        and len(results["metadatas"]) > 0
        and results["metadatas"][0] is not None
    ):
        # Convert Mapping to dict for consistent return type
        return dict(results["metadatas"][0])
    elif (
        results["documents"]
        and len(results["documents"]) > 0
        and results["documents"][0] is not None
    ):
        doc_content_str = results["documents"][0]
        try:
            loaded_json = json.loads(doc_content_str)
            if isinstance(loaded_json, dict):
                return loaded_json
            return {"description": doc_content_str}  # If not a dict, wrap it
        except json.JSONDecodeError:
            return {"description": doc_content_str}
    return {}


def keyword_search_documentation(query: str) -> list[SearchResult]:
    """
    Perform keyword search on documentation using token matching.

    Args:
        query: The query to search for.

    Returns:
        A list of search results from keyword matching.
    """
    n_results = 10
    collection = get_doc_collection()

    pattern = r"[ ,.!?\-_=|]+"
    query_tokens = [
        token.strip() for token in re.split(pattern, query) if token.strip()
    ]
    if len(query_tokens) > 1:
        where_document = {"$or": [{"$contains": token} for token in query_tokens]}
    else:
        where_document = {"$contains": query_tokens[0]}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where_document=cast(WhereDocument, where_document),
        include=[IncludeEnum.documents, IncludeEnum.metadatas, IncludeEnum.distances],
    )

    search_results = []
    if (
        results["documents"]
        and results["ids"]
        and results["ids"][0]
        and len(results["documents"][0]) > 0
    ):
        for i, doc in enumerate(results["documents"][0]):
            current_metadata = None
            if (
                results["metadatas"]
                and results["metadatas"][0]
                and i < len(results["metadatas"][0])
            ):
                current_metadata = results["metadatas"][0][i]
            search_results.append(
                SearchResult(
                    id=results["ids"][0][i],
                    content=doc,
                    metadata=current_metadata,
                )
            )
    return search_results


def search_examples(query: str) -> list[SearchResult]:
    """
    Search the examples for the given query string.

    Args:
        query: The query to search for.
        n_results: The number of results to return.

    Returns:
        A tuple of the ids and documents that match the query.
    """
    res = get_example_collection().query(query_texts=[query], n_results=2)
    if len(res["ids"]) == 0 or res["documents"] is None:
        return []
    return [
        SearchResult(
            id=res["ids"][0][i],
            content=res["documents"][0][i],
        )
        for i in range(len(res["ids"][0]))
    ]


"""
Workflow Tool:
You have access to a powerful tool called "workflow_tool". This tool allows 
you to design new workflows for the user.

Here's how to use it:

1. When a user requests a new workflow or you identify an opportunity to 
   create one, design the workflow using your knowledge of Nodetool nodes 
   and their connections.

2. Structure the workflow as a JSON object with the following properties:
   - name: A descriptive name for the workflow
   - description: A brief explanation of what the workflow does
   - graph: An object containing two arrays:
     - nodes: Each node should have an id, type, data (properties), and 
              ui_properties
     - edges: Connections between nodes, each with an id, source, target, 
              sourceHandle, and targetHandle

3. Make sure all nodes are connected properly and the workflow is logically
    sound. Important: Only use existing Nodetool nodes in the workflow.

4. Call the "workflow_tool" with this JSON object as its parameter.

This feature allows you to not only suggest workflows but actually implement 
them, greatly enhancing your ability to assist users. Be creative in 
designing workflows that solve user problems or demonstrate Nodetool 
capabilities.

Example usage:
User: "Can you create a workflow that generates an image and then applies a 
sepia filter?"
You: "Yes, here it is:"

Then proceed to design the workflow by calling the tool with the name, description
and graph properties, including all necessary nodes and edges.
"""


CORE_DOCS = [
    {
        "id": "models",
        "title": "Models",
        "content": """
        Local Models:
        - Local models need a GPU or MPS to run fast, smaller models can run on CPU
        - Model files can be large, please check your disk space before downloading
        - Remote models require API keys, you can set them in the settings menu

        Remote Models:
        - Remote API Providers require an account and API keys
        - Fal.ai gives access to a wide range of image and video models
        - Replicate gives access to a wide range of models
        - OpenAI and Anthropic models give access to worlds' most powerful language models
        """,
    },
    {
        "id": "assets",
        "title": "Assets",
        "content": """
        Assets:
        - Assets are either uploaded by the user or generated by nodes
        - Drag images, videos, audio, text, or any other files (from FileExplorer / Finder) onto the Asset panel on the right to import them
        - Drag images, videos, audio, text, or any other files onto the canvas to create constant asset nodes
        - Double-click on any asset in a node or inside the ASSETS panel to open it in the AssetViewer
        - Right-click on any asset to open the Asset Menu for more options
        - Select multiple assets by holding CTRL or SHIFT
        - Move assets between folders by dragging them onto the desired folder,
        - or use the right click menu for moving them into nested folders
        - Search for assets by typing in the search bar
        - Sort assets by clicking on the name or date buttons
        - Download: select one or more assets and use the right click menu
        - Delete: right click menu or X button
        - Rename: right click menu or press F2 key (also works with multiple assets)
        """,
    },
    {
        "id": "workflow_basics",
        "title": "Workflow Basics",
        "content": """
        ## Creating Workflows
        - Start with an empty canvas in the workflow editor
        - Add nodes by double-clicking or using CTRL+Space
        - Connect nodes by dragging from output to input handles
        - Configure node parameters in the right panel
        - Save your workflow using the save button
        - Run workflows with the play button
        
        ## Best Practices
        - Name your nodes descriptively
        - Group related nodes together
        - Test workflows incrementally
        - Use comments to document complex parts
        - Back up important workflows
        """,
    },
    {
        "id": "keyboard_shortcuts",
        "title": "Keyboard Shortcuts",
        "content": """
        ## Essential Shortcuts
        - Node Menu: Double-click canvas or Space
        - Run Workflow: CTRL+Enter / Cmd+Enter
        - Stop Workflow: ESC
        - Save: CTRL+S / Cmd+S
        - Undo/Redo: CTRL+Z / Cmd+Z
        - Delete Node: Delete or Backspace
        - Copy/Paste: CTRL+C / CTRL+V / Cmd+C / Cmd+V
        - Select All: CTRL+A / Cmd+A
        - Copy selected nodes: Shift + C and Paste selected nodes with Shift + V
        - Select multiple Nodes: Drag area with left click, Shift + Left Click if using LMB for panning
        - Select multiple nodes: click on nodes with CTRL key or draw a rectangle around nodes
        """,
    },
    {
        "id": "troubleshooting",
        "title": "Troubleshooting",
        "content": """
        ## Common Issues
        - Check model requirements before downloading
        - Verify API keys are correctly set for remote services
        - Ensure sufficient disk space for models
        - Monitor GPU memory usage
        - Check node connections for errors
        
        ## Getting Help
        - Use the help menu (? icon)
        - Check documentation
        - Visit the community forum
        - Report bugs through GitHub
        """,
    },
]


def index_core_docs(collection: chromadb.Collection):
    collection.add(
        ids=[doc["id"] for doc in CORE_DOCS],
        documents=[doc["content"] for doc in CORE_DOCS],
        metadatas=[{"title": doc["title"], "id": doc["id"]} for doc in CORE_DOCS],
    )


SYSTEM_PROMPT = """
You're an AI assistant for Nodetool, a no-code AI workflow platform. 
YOU ARE CONFIDENT AND KNOWLEDGEABLE.
DO NOT QUESTION YOURSELF.
        
NodeTool enables you to create custom AI workflows on your computer.

## Features ✨
- **Visual Editor**: 
  Create AI workflows visually without coding.
  A workflow is a graph of nodes.
  Each node has a name, type, and a set of parameters.
  Nodes can have multiple inputs and outputs.
  The node editor is a canvas that you can use to create your workflows.
  You can connect nodes by dragging from output to input handles.
  You can configure node parameters in the node itself.
  The workflow is executed by evaluating the graph from start to end.
  You can run the workflow by clicking the play button.
  Nodes can take strings, numbers, images, audio, video, and documents as input or output.
  One node is like a python function, it takes input, does something, and produces output.
  Many nodes run AI models, for example a text generation node runs a language model.
- **Local Models**: 
  Run models on your hardware.
  Nodetool's model manager can download models from Hugging Face and other sources.
  AI models need a NVidia GPU or Apple MPS to run fast.
  Nodetool offers many libraries for working with data, images, audio, video, and more.
  For example, image can be edited, audio can be transcribed, and documents can be summarized.
- **Integration with AI Platforms**:
  For more procesing power you can use remote models from OpenAI, Hugging Face, Ollama, Replicate, ElevenLabs, Google, Anthropic, and more.
- **Asset Browser**: 
  Import and manage media assets.
  The assets can be used as input or output for nodes.
  
IMPORTANT NODES:
- Preview Node: Renders any data as a preview, like images, videos, audio, documents, and more.
- Input Nodes: These nodes take user input, like text, images, audio, video, and more.
- Chat Input Node: This node takes user input from a chat interface, including audio, image or documents.
- Constant Node: This node takes a constant value as input, like a string, number, or image.
- Output Node: This node takes any data as input and displays it to the user.
- Loop Node: This node takes list or dataframes and applies a sub graph to each element of the list.
- Text Generation: There are many nodes from different providers for text generation, like OpenAI, Ollama, Google, Anthropic, and more.
- Image Generation: There are many nodes from different providers for image generation, like OpenAI, Hugging Face, Replicate, and more.

## Use Cases 🎨
- 🎨 **Personal Learning Assistant**: Create chatbots that read and explain your PDFs, e-books, or academic papers
- 📝 **Note Summarization**: Extract key insights from Obsidian or Apple Notes
- 🎤 **Voice Memo to Presentation**: Convert recorded ideas into documents
- 🔧️ **Image Generation & Editing**: Create and modify images with advanced AI models
- 🎵 **Audio Processing**: Generate and edit audio content with AI assistance
- 🎬 **Video Creation**: Produce and manipulate video content using AI tools
- ⚡ **Automation**: Streamline repetitive tasks with AI-powered scripts

Key Guidelines:
- **Reference Valid Nodes:** When mentioning a node, only reference existing ones. Use the format [Node Type](/help/node_type) for clarity.
- **Use Documentation Search:** Use the documentation search tool to find information about nodes.
- **Use Example Search:** Use the example search tool to find examples of how to use nodes.
- **Answer Precisely:** Be concise, clear, and creative in your responses. Utilize ASCII diagrams if they help explain complex workflows.
- **Focus on Nodetool Features:** Emphasize the visual editor, asset management, model management, workflow execution, and keyboard shortcuts (for example, the help menu in the top right corner).

HOW TO USE SEARCH TOOLS:

1. Semantic Search Documentation:
   - Use semantic_search_documentation() for meaning-based searches
   - Best for conceptual queries and finding related content
   - Example queries:
     - "How to generate images?" -> semantic_search_documentation("image generation")
     - "What nodes can generate text?" -> semantic_search_documentation("text generation")
   - Parameters:
     - query: str - Your search query

2. Keyword Search Documentation:
   - Use keyword_search_documentation() for exact word matches
   - Best for finding specific node types or features
   - Example queries:
     - "What is GPT?" -> keyword_search_documentation("GPT")
     - "How to use Pandas?" -> keyword_search_documentation("Pandas")
   - Parameters:
     - query: str - Your search query

3. Example Search:
   - Use search_examples() to find relevant workflow examples
   - Best for finding example workflows and use cases
   - Example queries:
     - "How to build a chatbot?" -> search_examples("chatbot")
     - "How to build a text to speech workflow?" -> search_examples("text to speech")
   - Parameters:
     - query: str - Your search query
     
4. Node Properties:
   - Use node_properties() to get the properties of a node
   - Best for finding node properties and use cases
   - Example queries:
     - "What are the inputs of the Ollama node?" -> node_properties("Ollama")
     - "What is the output of the ImagePreview node?" -> node_properties("ImagePreview")
   - Parameters:
     - node_type: str - The type of the node

When a user asks a question:
1. First try semantic search to understand the topic
2. If looking for specific nodes/features, use keyword search
3. For implementation examples, use example search
4. For details about a node, use node_properties
5. Combine the results to provide a comprehensive answer

REFERENCE NODES:
- Format any reference to a node as: [Node Type](/help/node_type)
- Example node link: [Text Generation](/help/huggingface.text.TextGeneration)
- DO NOT ADD http/domain to URLs.

HOW TO ANSWER QUESTIONS:
- Explain any necessary Nodetool features
- KEEP IT BRIEF
- DO NOT OVERTHINK
- BE CONCISE
"""


import PIL.Image


async def create_message(message: Message) -> Mapping[str, str | list[str]]:
    ollama_message: dict[str, str | list[str]] = {
        "role": message.role,
    }

    if isinstance(message.content, list):
        ollama_message["content"] = "\n".join(
            content.text
            for content in message.content
            if isinstance(content, MessageTextContent)
        )
    else:
        ollama_message["content"] = str(message.content)

    return ollama_message


# --- Pydantic Models for Tool Arguments ---
class SearchArgs(BaseModel):
    query: str


class NodePropertiesArgs(BaseModel):
    node_type: str


# --- Tool Classes for Help System ---
class BaseHelpTool(Tool):
    _args_model_class: Type[BaseModel]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "_args_model_class") and self._args_model_class:
            self.input_schema = self._args_model_class.model_json_schema()
        else:
            self.input_schema = super().input_schema

    # Override process to parse params with Pydantic model
    async def _process_typed(
        self, context: ProcessingContext, typed_params: BaseModel
    ) -> Any:
        # This method should be implemented by subclasses if they want to work with typed_params
        raise NotImplementedError(
            "Subclasses must implement _process_typed or override process directly"
        )

    async def process(self, context: ProcessingContext, params: Dict[str, Any]) -> Any:
        # Parse the raw params dict using the specific Pydantic model for this tool
        if hasattr(self, "_args_model_class") and self._args_model_class:
            try:
                typed_params_instance = self._args_model_class(**params)
                return await self._process_typed(context, typed_params_instance)
            except Exception as e:  # Catch Pydantic validation errors or other issues
                log.error(
                    f"Error parsing params for tool {self.name} with model {self._args_model_class.__name__}: {params}. Error: {e}"
                )
                # Return a JSON string error, as tool results are often expected to be strings by LLMs
                return json.dumps(
                    {"error": f"Invalid parameters for tool {self.name}: {str(e)}"}
                )
        else:
            log.warning(
                f"Tool {self.name} is missing _args_model_class for Pydantic validation."
            )
            return await super().process(
                context, params
            )  # Fallback to base if no model defined


class SemanticSearchDocumentationTool(BaseHelpTool):
    name: str = "semantic_search_documentation"
    description: str = (
        "Performs semantic search on Nodetool documentation. Use for conceptual queries and finding related content."
    )
    _args_model_class: Type[BaseModel] = SearchArgs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _process_typed(self, context: ProcessingContext, typed_params: SearchArgs) -> str:  # type: ignore
        log.info(
            f"Executing SemanticSearchDocumentationTool with query: {typed_params.query}"
        )
        results = semantic_search_documentation(query=typed_params.query)
        return convert_results_to_json(results)


class KeywordSearchDocumentationTool(BaseHelpTool):
    name: str = "keyword_search_documentation"
    description: str = (
        "Performs keyword search on Nodetool documentation. Use for finding specific node types or features by exact word matches."
    )
    _args_model_class: Type[BaseModel] = SearchArgs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _process_typed(self, context: ProcessingContext, typed_params: SearchArgs) -> str:  # type: ignore
        log.info(
            f"Executing KeywordSearchDocumentationTool with query: {typed_params.query}"
        )
        results = keyword_search_documentation(query=typed_params.query)
        return convert_results_to_json(results)


class SearchExamplesTool(BaseHelpTool):
    name: str = "search_examples"
    description: str = (
        "Searches for relevant Nodetool workflow examples. Use for finding example workflows and use cases."
    )
    _args_model_class: Type[BaseModel] = SearchArgs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _process_typed(self, context: ProcessingContext, typed_params: SearchArgs) -> str:  # type: ignore
        log.info(f"Executing SearchExamplesTool with query: {typed_params.query}")
        results = search_examples(query=typed_params.query)
        return convert_results_to_json(results)


class NodePropertiesTool(BaseHelpTool):
    name: str = "node_properties"
    description: str = (
        "Gets the properties (inputs, outputs, description) of a specific Nodetool node type."
    )
    _args_model_class: Type[BaseModel] = NodePropertiesArgs

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _process_typed(self, context: ProcessingContext, typed_params: NodePropertiesArgs) -> str:  # type: ignore
        log.info(
            f"Executing NodePropertiesTool with node_type: {typed_params.node_type}"
        )
        properties = node_properties(node_type=typed_params.node_type)
        return convert_results_to_json(properties)


def convert_results_to_json(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json()
    elif isinstance(obj, list):
        return json.dumps(
            [
                (
                    json.loads(convert_results_to_json(item))
                    if isinstance(convert_results_to_json(item), str)
                    else convert_results_to_json(item)
                )
                for item in obj
            ]
        )
    elif isinstance(obj, dict):
        return json.dumps(
            {
                k: (
                    json.loads(convert_results_to_json(v))
                    if isinstance(convert_results_to_json(v), str)
                    else convert_results_to_json(v)
                )
                for k, v in obj.items()
            }
        )
    elif isinstance(obj, (str, int, float, bool)):
        return json.dumps(obj)
    else:
        try:
            return json.dumps(obj)
        except TypeError:
            return json.dumps(str(obj))


async def create_help_answer(
    provider: ChatProvider, messages: List[Message], model: str
) -> AsyncGenerator[Chunk | ToolCall, None]:
    """
    Generates help answers using a ChatProvider and a set of help-specific tools.
    Streams text chunks of the answer.
    """
    semantic_search_tool = SemanticSearchDocumentationTool()
    keyword_search_tool = KeywordSearchDocumentationTool()
    search_examples_tool = SearchExamplesTool()
    node_properties_tool = NodePropertiesTool()

    help_tools_instances: List[Tool] = [
        semantic_search_tool,
        keyword_search_tool,
        search_examples_tool,
        node_properties_tool,
    ]

    effective_messages_for_provider: List[Message] = [
        Message(role="system", content=SYSTEM_PROMPT)
    ] + messages

    dummy_processing_context = ProcessingContext()  # Create a dummy context

    while True:
        tool_messages_generated_this_iteration: List[Message] = []
        made_tool_call_this_iteration = False

        async for item in provider.generate_messages(
            messages=effective_messages_for_provider,
            model=model,
            tools=help_tools_instances,
        ):  # type: ignore
            if isinstance(item, Chunk):
                yield item
            elif isinstance(item, ToolCall):
                log.info(
                    f"Help system received tool call: {item.name} with args {item.args}"
                )
                yield item

                found_tool_instance = next(
                    (t for t in help_tools_instances if t.name == item.name), None
                )

                tool_call_id = item.id or str(uuid.uuid4())

                if found_tool_instance and isinstance(
                    found_tool_instance, BaseHelpTool
                ):
                    tool_args_dict = item.args if isinstance(item.args, dict) else {}
                    if not isinstance(item.args, dict):
                        log.warning(
                            f"Tool call arguments for {item.name} is not a dict: {item.args}. Attempting to use empty dict."
                        )

                    # The process method of BaseHelpTool will handle Pydantic parsing internally
                    tool_result_data = await found_tool_instance.process(
                        context=dummy_processing_context,  # Pass dummy context
                        params=tool_args_dict,  # Pass raw dict as params
                    )

                    assistant_tool_call_msg = Message(
                        role="assistant", tool_calls=[item]
                    )

                    tool_result_msg = Message(
                        role="tool",
                        tool_call_id=tool_call_id,
                        name=item.name,
                        content=tool_result_data,
                    )
                    tool_messages_generated_this_iteration.extend(
                        [assistant_tool_call_msg, tool_result_msg]
                    )
                    made_tool_call_this_iteration = True
                else:
                    raise ValueError(
                        f"Tool '{item.name}' not found or not a BaseHelpTool instance."
                    )

        if made_tool_call_this_iteration:
            effective_messages_for_provider.extend(
                tool_messages_generated_this_iteration
            )
        else:
            break


async def test_chat(provider: ChatProvider, model: str):
    """Simple terminal-based chat tester with readline support"""
    print("Starting help chat test (type 'exit' to quit)")
    print("This test uses the refactored create_help_answer with a provider.")

    try:
        doc_collection = get_doc_collection()
        if doc_collection.count() == 0:
            print("Indexing core documentation for test...")
            index_core_docs(doc_collection)
            print("Indexing package documentation for test...")
            index_documentation(doc_collection)

        example_collection = get_example_collection()
        if example_collection.count() == 0:
            print("Indexing examples for test...")
            index_examples(example_collection)
    except Exception as e:
        print(f"Error initializing ChromaDB collections for test: {e}")
        print("Searches might not work correctly.")

    chat_history: List[Message] = []

    readline.parse_and_bind("tab: complete")
    readline.set_completer(lambda text, state: None)

    print(f"Using model: {model}")

    while True:
        try:
            user_input = await asyncio.to_thread(input, "\n> ")
            if user_input.lower() == "exit":
                break
            if not user_input.strip():
                continue

            chat_history.append(Message(role="user", content=user_input))

            full_response = []
            print("Assistant: ", end="", flush=True)
            async for chunk in create_help_answer(
                provider=provider, messages=chat_history, model=model
            ):
                print(chunk, end="", flush=True)
                full_response.append(chunk)
            print()

            if full_response:
                chat_history.append(
                    Message(role="assistant", content="".join(full_response))
                )

        except KeyboardInterrupt:
            print("\nUse 'exit' to quit.")
            continue
        except EOFError:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nAn error occurred in test_chat: {e}")


if __name__ == "__main__":
    print("Initializing ChromaDB collections for __main__...")
    try:
        doc_collection = get_doc_collection()
        if doc_collection.count() == 0:
            print("Indexing core documentation...")
            index_core_docs(doc_collection)
            print("Indexing package documentation...")
            index_documentation(doc_collection)
        else:
            print("Documentation collections appear to be already indexed.")

        example_collection = get_example_collection()
        if example_collection.count() == 0:
            print("Indexing examples...")
            index_examples(example_collection)
        else:
            print("Example collection appears to be already indexed.")
        print("ChromaDB initialization complete.")
    except Exception as e:
        print(f"Error during __main__ ChromaDB initialization: {e}")
        print("Help tool searches might fail or return empty results.")

    asyncio.run(test_chat(provider=get_provider(Provider.OpenAI), model="gpt-4o-mini"))
