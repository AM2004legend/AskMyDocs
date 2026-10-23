"""One chat() function over several OpenAI-compatible providers (Gemini first, Groq as fallback).

Failure handling:
- 429 rate limit / 5xx / timeout -> try the next provider; if all are busy, back off
  (exponential, or the server's Retry-After) and go round again, up to llm_max_retries rounds.
- "request too large" -> raise LLMRequestTooLarge so the caller can split the work.
- bad key / unknown model -> try the next provider, otherwise raise a clear LLMError.
API keys are never put in error messages or logs.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import openai

from app.config import get_settings

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Something went wrong talking to the LLM; the message is safe to show the user."""


class LLMNotConfigured(LLMError):
    pass


class LLMRequestTooLarge(LLMError):
    pass


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str


TOO_LARGE_HINTS = ("too large", "context length", "context_length", "maximum context", "reduce the length")


def _is_too_large(exc: openai.APIStatusError) -> bool:
    return exc.status_code == 413 or (exc.status_code == 400 and any(h in str(exc).lower() for h in TOO_LARGE_HINTS))


def _retry_after(exc: openai.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None


class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        max_retries: int = 4,
        timeout: float = 120.0,
        min_interval: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        make_client: Callable[[Provider], object] | None = None,
    ):
        self.providers = providers
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._sleep = sleep
        self._last_call = 0.0
        self._pace_lock = threading.Lock()
        make_client = make_client or (
            lambda p: openai.OpenAI(base_url=p.base_url, api_key=p.api_key, max_retries=0, timeout=timeout)
        )
        self._clients = {p.name: make_client(p) for p in providers}

    @property
    def configured(self) -> bool:
        return bool(self.providers)

    def _pace(self) -> None:
        """Keep at least min_interval seconds between paced calls (free tiers count requests/minute)."""
        with self._pace_lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                self._sleep(wait)
            self._last_call = time.monotonic()

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int | None = None,
        paced: bool = False,
    ) -> str:
        if not self.providers:
            raise LLMNotConfigured("No LLM is configured. Set LLM_API_KEY and LLM_MODEL in .env.")

        last_problem = "unknown error"
        for attempt in range(self.max_retries + 1):
            transient = False  # did any provider fail in a way that waiting could fix?
            wait = min(2.0 ** (attempt + 1), 60.0)
            for provider in self.providers:
                if paced:
                    self._pace()
                kwargs = {"model": provider.model, "messages": messages, "temperature": temperature}
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                try:
                    response = self._clients[provider.name].chat.completions.create(**kwargs)
                    return (response.choices[0].message.content or "").strip()
                except openai.RateLimitError as exc:
                    transient, last_problem = True, f"{provider.name} rate limit"
                    wait = max(wait, min(_retry_after(exc) or 0.0, 60.0))
                    log.warning("LLM %s rate-limited (attempt %d)", provider.name, attempt + 1)
                except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                    transient, last_problem = True, f"{provider.name} unavailable"
                    log.warning("LLM %s unavailable (attempt %d)", provider.name, attempt + 1)
                except openai.APIStatusError as exc:
                    if _is_too_large(exc):
                        raise LLMRequestTooLarge("The request was too large for the model.") from exc
                    # bad key / unknown model: waiting won't fix it, but the next provider might work
                    last_problem = f"{provider.name} returned HTTP {exc.status_code}"
                    log.error("LLM %s failed with HTTP %s", provider.name, exc.status_code)
            if not transient:
                break
            if attempt < self.max_retries:
                self._sleep(wait)

        if "rate limit" in last_problem:
            raise LLMError("The AI service is busy (rate limit reached). Please wait a minute and try again.")
        raise LLMError(f"Could not get an answer from the AI service ({last_problem}).")


_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        s = get_settings()
        providers = []
        if s.llm_api_key.get_secret_value() and s.llm_model:
            providers.append(Provider("primary", s.llm_base_url, s.llm_api_key.get_secret_value(), s.llm_model))
        if s.fallback_llm_api_key.get_secret_value() and s.fallback_llm_model:
            providers.append(
                Provider(
                    "fallback", s.fallback_llm_base_url, s.fallback_llm_api_key.get_secret_value(), s.fallback_llm_model
                )
            )
        _client = LLMClient(
            providers,
            max_retries=s.llm_max_retries,
            timeout=s.llm_timeout_seconds,
            min_interval=s.llm_min_interval_seconds,
        )
    return _client
