=====
Usage
=====

NodeTool is an open-source platform designed to simplify the creation and deployment of AI applications without requiring coding expertise.

Getting Started
--------------

To use NodeTool in a project:

.. code-block:: python

    import nodetool

Visual Workflow Editor
---------------------

NodeTool provides an intuitive visual workflow editor that allows you to build AI applications without writing code:

1. Create a new workflow in the NodeTool Editor
2. Drag and drop nodes from the node palette
3. Connect nodes to define the flow of data
4. Configure node parameters
5. Test and deploy your workflow

Available Node Types
-------------------

NodeTool supports a wide range of node types:

- **Anthropic** 🧠: Text-based AI tasks
- **Comfy** 🎨: Support for ComfyUI nodes for image processing
- **Chroma** 🌈: Vector database for embeddings
- **ElevenLabs** 🎤: Text-to-speech services
- **Fal** 🔊: AI for audio, image, text, and video
- **Google** 🔍: Access to Gemini Models and Gmail
- **HuggingFace** 🤗: AI for audio, image, text, and video
- **NodeTool Core** ⚙️: Core data and media processing functions
- **Ollama** 🦙: Run local language models
- **OpenAI** 🌐: AI for audio, image, and text tasks
- **Replicate** ☁️: AI for audio, image, text, and video in the cloud

Implementing Custom Nodes
------------------------

Extend NodeTool's functionality by creating custom nodes:

.. code-block:: python

    class MyAgent(BaseNode):
        prompt: Field(default="Build me a website for my business.")

        async def process(self, context: ProcessingContext) -> str:
            llm = MyLLM()
            return llm.generate(self.prompt)
