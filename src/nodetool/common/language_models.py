#!/usr/bin/env python

from nodetool.metadata.types import LanguageModel, Provider

anthropic_models = [
    LanguageModel(
        id="claude-3-5-haiku-latest",
        name="Claude 3.5 Haiku",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-3-5-sonnet-latest",
        name="Claude 3.5 Sonnet",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-3-7-sonnet-latest",
        name="Claude 3.7 Sonnet",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-sonnet-4-20250514",
        name="Claude Sonnet 4",
        provider=Provider.Anthropic,
    ),
    LanguageModel(
        id="claude-opus-4-20250514",
        name="Claude Opus 4",
        provider=Provider.Anthropic,
    ),
]

gemini_models = [
    LanguageModel(
        id="gemini-2.5-pro-exp-03-25",
        name="Gemini 2.5 Pro Experimental",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.5-flash-preview-04-17",
        name="Gemini 2.5 Flash",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash-lite",
        name="Gemini 2.0 Flash Lite",
        provider=Provider.Gemini,
    ),
    LanguageModel(
        id="gemini-2.0-flash-exp-image-generation",
        name="Gemini 2.0 Flash Exp Image Generation",
        provider=Provider.Gemini,
    ),
]

openai_models = [
    LanguageModel(
        id="codex-mini-latest",
        name="Codex Mini",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o",
        name="GPT-4o",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-audio-preview-2024-12-17",
        name="GPT-4o Audio",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-mini",
        name="GPT-4o Mini",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4o-mini-audio-preview-2024-12-17",
        name="GPT-4o Mini Audio",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="chatgpt-4o-latest",
        name="ChatGPT-4o",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4.1",
        name="GPT-4.1",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="gpt-4.1-mini",
        name="GPT-4.1 Mini",
        provider=Provider.OpenAI,
    ),
    LanguageModel(
        id="o4-mini",
        name="O4 Mini",
        provider=Provider.OpenAI,
    ),
]

huggingface_models = [
    LanguageModel(
        id="HuggingFaceTB/SmolLM3-3B",
        name="SmolLM3 3B",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="deepseek-ai/DeepSeek-V3-0324",
        name="DeepSeek V3 0324",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="tngtech/DeepSeek-TNG-R1T2-Chimera",
        name="DeepSeek TNG R1T2 Chimera",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="tencent/Hunyuan-A13B-Instruct",
        name="Hunyuan A13B Instruct",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="agentica-org/DeepSWE-Preview",
        name="DeepSWE Preview",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="google/gemma-2-2b-it",
        name="Gemma 2 2B IT",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        name="DeepSeek R1 Distill Qwen 1.5B",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        name="Meta Llama 3.1 8B Instruct",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="microsoft/phi-4",
        name="Phi 4",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="Qwen/Qwen2.5-7B-Instruct-1M",
        name="Qwen 2.5 7B Instruct 1M",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="Qwen/Qwen2.5-Coder-32B-Instruct",
        name="Qwen 2.5 Coder 32B Instruct",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="deepseek-ai/DeepSeek-R1",
        name="DeepSeek R1",
        provider=Provider.HuggingFace,
    ),
    LanguageModel(
        id="Qwen/Qwen2.5-VL-7B-Instruct",
        name="Qwen 2.5 VL 7B Instruct",
        provider=Provider.HuggingFace,
    ),
]

huggingface_groq_models = [
    LanguageModel(
        id="meta-llama/Meta-Llama-3-70B-Instruct",
        name="Meta Llama 3 70B Instruct",
        provider=Provider.HuggingFaceGroq,
    ),
    LanguageModel(
        id="meta-llama/Llama-3.3-70B-Instruct",
        name="Llama 3.3 70B Instruct",
        provider=Provider.HuggingFaceGroq,
    ),
    LanguageModel(
        id="meta-llama/Llama-Guard-4-12B",
        name="Llama Guard 4 12B",
        provider=Provider.HuggingFaceGroq,
    ),
    LanguageModel(
        id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        name="Llama 4 Scout 17B 16E Instruct",
        provider=Provider.HuggingFaceGroq,
    ),
    LanguageModel(
        id="meta-llama/Llama-4-Maverick-17B-128E-Instruct",
        name="Llama 4 Maverick 17B 128E Instruct",
        provider=Provider.HuggingFaceGroq,
    ),
]

huggingface_cerebras_models = [
    LanguageModel(
        id="cerebras/Cerebras-GPT-2.5-12B-Instruct",
        name="Cerebras GPT 2.5 12B Instruct",
        provider=Provider.HuggingFaceCerebras,
    ),
    LanguageModel(
        id="meta-llama/Llama-3.3-70B-Instruct",
        name="Llama 3.3 70B Instruct",
        provider=Provider.HuggingFaceCerebras,
    ),
    LanguageModel(
        id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        name="Llama 4 Scout 17B 16E Instruct",
        provider=Provider.HuggingFaceCerebras,
    ),
]