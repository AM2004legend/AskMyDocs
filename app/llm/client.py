"""One chat() function over several OpenAI-compatible models (Gemini first, Groq as fallback).

Built around what the free tiers actually allow (measured 2026-10-24):
- Gemini free: ~20 requests/day *per model*  -> list several models; each has its own quota.
- Groq free:   ~1000 requests/day, but only 8000 tokens/minute -> fine for many small calls.

Failure handling, per request:
- 429 per-minute limit / 5xx / timeout -> try the next model; if all are busy, back off
  (exponential, or the server's Retry-After) and go round again, up to llm_max_retries rounds.
- 429 *daily* quota -> that model is skipped until its quota resets (no pointless retries).
- "request too large" -> skip that model for this request; only if every model says so,
  raise LLMRequestTooLarge so the caller can split the work.
- bad key / unknown model -> skip that model; raise a clear LLMError if nothing works.
Background work (paced=True, e.g. the document map) tries the fallback first, keeping the
scarce primary quota for interactive answers. API keys never appear in messages or logs.
"""

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import openai

from app.config import get_settings
from app.privacy import tripwire

log = logging.getLogger(__name__)


class LLMError(Exception):
    """Something went wrong talking to the LLM; the message is safe to show the user."""


class LLMNotConfigured(LLMError):
    pass


class LLMRequestTooLarge(LLMError):
    pass


class LLMBlockedPII(LLMError):
    """The tripwire found raw PII in a request for a sensitive document."""


@dataclass
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    role: str = "primary"  # "primary" or "fallback"
    reasoning_effort: str = ""  # "low" for reasoning models, "" = provider default


TOO_LARGE_HINTS = ("too large", "context length", "context_length", "maximum context", "reduce the length")


def _is_too_large(exc: openai.APIStatusError) -> bool:
    return exc.status_code == 413 or (exc.status_code == 400 and any(h in str(exc).lower() for h in TOO_LARGE_HINTS))


def _retry_after(exc: openai.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after"))
    except (TypeError, ValueError, AttributeError):
        return None


def _daily_quota_reset(exc: openai.APIStatusError) -> float | None:
    """Seconds until a *daily* quota resets, or None if this 429 is a short per-minute limit."""
    message = str(exc)
    if "perday" not in message.lower().replace(" ", ""):
        return None
    delay = re.search(r"retryDelay'?\"?:\s*'?\"?(\d+)s", message)
    return float(delay.group(1)) if delay else 3600.0


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
        self._exhausted_until: dict[str, float] = {}  # model -> monotonic time its daily quota resets
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

    def _order(self, paced: bool) -> list[Provider]:
        if not paced:
            return self.providers
        return sorted(self.providers, key=lambda p: p.role != "fallback")  # background work: fallback first

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: int | None = None,
        paced: bool = False,
    ) -> str:
        if not self.providers:
            raise LLMNotConfigured("No LLM is configured. Set LLM_API_KEY and LLM_MODEL in .env.")
        if tripwire.is_active():  # working on a sensitive document: nothing raw may leave
            found = tripwire.scan_payload(messages)
            if found:
                log.warning("Tripwire blocked an LLM request containing: %s", ", ".join(found))  # types only
                raise LLMBlockedPII(
                    f"Blocked: the request still contained personal data ({', '.join(found)}), so it was not sent."
                )

        last_problem = "unknown error"
        too_large: set[str] = set()  # models whose limits this request exceeds (e.g. Groq's 8k tokens/min)
        broken: set[str] = set()  # bad key / unknown model: don't call again this request
        for attempt in range(self.max_retries + 1):
            transient = False  # did any model fail in a way that waiting could fix?
            wait = min(2.0 ** (attempt + 1), 60.0)
            now = time.monotonic()
            usable = [
                p
                for p in self._order(paced)
                if p.name not in too_large | broken and self._exhausted_until.get(p.name, 0) <= now
            ]
            for provider in usable:
                if paced:
                    self._pace()
                kwargs = {"model": provider.model, "messages": messages, "temperature": temperature}
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                if provider.reasoning_effort:
                    kwargs["reasoning_effort"] = provider.reasoning_effort
                try:
                    response = self._clients[provider.name].chat.completions.create(**kwargs)
                    return (response.choices[0].message.content or "").strip()
                except openai.RateLimitError as exc:
                    reset = _daily_quota_reset(exc)
                    if reset is not None:
                        self._exhausted_until[provider.name] = time.monotonic() + reset
                        last_problem = "daily quota"
                        log.warning("LLM %s: daily free quota used up; skipping it for %.0f min", provider.name,
                                    reset / 60)  # fmt: skip
                        continue
                    transient, last_problem = True, f"{provider.name} rate limit"
                    wait = max(wait, min(_retry_after(exc) or 0.0, 60.0))
                    log.warning("LLM %s rate-limited (attempt %d)", provider.name, attempt + 1)
                except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError):
                    transient, last_problem = True, f"{provider.name} unavailable"
                    log.warning("LLM %s unavailable (attempt %d)", provider.name, attempt + 1)
                except openai.APIStatusError as exc:
                    if _is_too_large(exc):
                        # too big for THIS model; another one may still take it
                        too_large.add(provider.name)
                        log.warning("LLM %s: request too large", provider.name)
                        continue
                    broken.add(provider.name)
                    last_problem = f"{provider.name} returned HTTP {exc.status_code}"
                    log.error("LLM %s failed with HTTP %s", provider.name, exc.status_code)
            now = time.monotonic()
            still_usable = [
                p
                for p in self.providers
                if p.name not in too_large | broken and self._exhausted_until.get(p.name, 0) <= now
            ]
            if too_large and not still_usable:
                raise LLMRequestTooLarge("The request was too large for the available models.")
            if not transient:
                break
            if attempt < self.max_retries:
                self._sleep(wait)

        if last_problem == "daily quota" or (not transient and self._exhausted_until):
            raise LLMError(
                "The free daily AI quota is used up for every configured model. "
                "It resets within a day; adding a second model or provider in .env helps."
            )
        if "rate limit" in last_problem:
            raise LLMError("The AI service is busy (rate limit reached). Please wait a minute and try again.")
        raise LLMError(f"Could not get an answer from the AI service ({last_problem}).")


_client: LLMClient | None = None


def _models(value: str) -> list[str]:
    return [m.strip() for m in value.split(",") if m.strip()]


def get_llm() -> LLMClient:
    global _client
    if _client is None:
        s = get_settings()
        providers = []
        for role, base_url, key, models, effort in (
            ("primary", s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_reasoning_effort),
            ("fallback", s.fallback_llm_base_url, s.fallback_llm_api_key, s.fallback_llm_model,
             s.fallback_llm_reasoning_effort),
        ):  # fmt: skip
            if key.get_secret_value():
                for model in _models(models):  # comma-separated: each model has its own free quota
                    providers.append(Provider(f"{role}:{model}", base_url, key.get_secret_value(), model, role, effort))
        _client = LLMClient(
            providers,
            max_retries=s.llm_max_retries,
            timeout=s.llm_timeout_seconds,
            min_interval=s.llm_min_interval_seconds,
        )
    return _client
