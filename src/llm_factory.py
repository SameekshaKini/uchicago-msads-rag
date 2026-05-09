import os
import logging

log = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "azure_openai")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOY = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
LLM_TEMP = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))


def get_llm(provider: str = LLM_PROVIDER):
    provider = provider.lower()

    if provider == "azure_openai":
        log.info("Using Azure OpenAI LLM: %s", AZURE_DEPLOY)
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=AZURE_DEPLOY,
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_VERSION,
            temperature=LLM_TEMP,
            max_tokens=LLM_MAX_TOKENS,
        )

    elif provider == "openai":
        log.info("Using OpenAI LLM: gpt-4o-mini")
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=LLM_TEMP,
            max_tokens=LLM_MAX_TOKENS,
        )

    elif provider == "huggingface":
        log.info("Using HuggingFace LLM: %s", HF_LLM_MODEL)
        from langchain_huggingface import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            repo_id=HF_LLM_MODEL,
            huggingfacehub_api_token=HF_API_KEY or None,
            temperature=LLM_TEMP,
            max_new_tokens=LLM_MAX_TOKENS,
            task="text-generation",
        )

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. "
            "Choose 'azure_openai', 'openai', or 'huggingface'."
        )
