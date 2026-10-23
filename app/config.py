"""All settings in one place, read from environment variables / the .env file.

API keys are SecretStr so they never show up if settings get printed or logged.
"""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://askmydocs:askmydocs@localhost:5433/askmydocs"

    # primary LLM (Gemini by default) and optional fallback (Groq) used on rate limits
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = ""
    fallback_llm_base_url: str = "https://api.groq.com/openai/v1"
    fallback_llm_api_key: SecretStr = SecretStr("")
    fallback_llm_model: str = ""
    llm_max_retries: int = 4
    llm_timeout_seconds: float = 120.0
    llm_min_interval_seconds: float = 0.0  # spacing between calls while building the doc map

    # embeddings (run locally)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embed_dim: int = 384

    # chunking: MiniLM only reads the first 256 word-pieces (~190 words), so keep chunks below that
    chunk_words: int = 180
    chunk_overlap_words: int = 40

    # retrieval
    vector_top_k: int = 20
    keyword_top_k: int = 20
    final_top_k: int = 6
    rrf_k: int = 60
    min_vector_similarity: float = 0.25  # below this best-match cosine similarity we refuse

    # document map
    map_batch_words: int = 12000  # how many words of chapters go into one summarising call
    summary_words: int = 120

    # uploads + OCR
    max_upload_mb: int = 20
    ocr_enabled: bool = True
    ocr_min_chars: int = 25  # a PDF page with less extractable text than this gets OCR'd
    ocr_render_scale: float = 2.5  # 72 dpi * 2.5 = 180 dpi


@lru_cache
def get_settings() -> Settings:
    return Settings()
