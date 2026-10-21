"""Shared DeepSeek client. Standard library only.

Worldbox talks to DeepSeek-V4.1-Flash over HTTPS. Narration and tribe minds
both go through here so there is one place for auth, retries and TLS.

Keys come from ``.env`` or the environment: ``DEEPSEEK_API_KEY`` and
``OPENAI_API_KEY``. Either is enough; both together means tribes and
citizens get mixed houses, so the world is not one model talking to itself.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# DeepSeek-V4.1-Flash. The `deepseek-flash` id always points at the current
# Flash model; older aliases (deepseek-v4-flash, …) are routed here too.
DEFAULT_MODEL = "deepseek-flash"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
API_URL = "https://api.deepseek.com/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
ENV_KEY = "DEEPSEEK_API_KEY"
OPENAI_ENV_KEY = "OPENAI_API_KEY"

# Each house is one LLM the simulation can trap as a people.
HOUSES: tuple[dict[str, str], ...] = (
    {
        "house": "deepseek",
        "model": DEFAULT_MODEL,
        "label": "DeepSeek Flash",
        "env": ENV_KEY,
        "url": API_URL,
    },
    {
        "house": "openai",
        "model": DEFAULT_OPENAI_MODEL,
        "label": "GPT-5 Mini",
        "env": OPENAI_ENV_KEY,
        "url": OPENAI_URL,
    },
)

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
_DOTENV_LOADED = False


def load_dotenv() -> None:
    """Load KEY=VALUE lines from a local ``.env`` without overwriting the real env.

    Looks at the project root first, then the current directory and its
    parents. Existing environment variables win, so ``export DEEPSEEK_API_KEY``
    still overrides the file. The file is never required; a missing ``.env``
    is silence, not an error.
    """
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    _DOTENV_LOADED = True

    seen: set[Path] = set()
    candidates = [Path(__file__).resolve().parents[2] / ".env"]
    cwd = Path.cwd()
    candidates.extend(directory / ".env" for directory in (cwd, *cwd.parents))
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            if "=" not in line:
                continue
            name, value = line.split("=", 1)
            name = name.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if name and value and name not in os.environ:
                os.environ[name] = value


def api_key(explicit: Optional[str] = None, env_name: str = ENV_KEY) -> Optional[str]:
    """The API key to use, from the argument, the environment, or ``.env``."""
    if explicit:
        return explicit
    load_dotenv()
    return os.environ.get(env_name) or None


# Any OpenAI-compatible endpoint can be one more house, configured from .env:
#   LLM_BASE_URL  e.g. https://api.groq.com/openai/v1, or http://127.0.0.1:11434/v1
#   LLM_MODEL     the provider's model name
#   LLM_API_KEY   optional -- a local server like Ollama needs none
#   LLM_LABEL     optional display name
# LLM_BASE_URL=mock selects the offline rule-based mind (ai/mock.py) instead.
CUSTOM_ENV_KEY = "LLM_API_KEY"
MOCK_TOKENS = frozenset({"mock", "offline"})


def _custom_spec() -> Optional[Dict[str, str]]:
    """The .env-configured house, if any."""
    load_dotenv()
    base = (os.environ.get("LLM_BASE_URL") or "").strip()
    if base.lower() in MOCK_TOKENS:
        return {"house": "mock", "model": "mock-mind", "label": "Mock mind (offline)",
                "env": "", "url": "", "keyless": "1"}
    model = (os.environ.get("LLM_MODEL") or "").strip()
    if not base or not model:
        return None
    url = base.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    return {
        "house": "custom",
        "model": model,
        "label": (os.environ.get("LLM_LABEL") or "").strip() or model,
        "env": CUSTOM_ENV_KEY,
        "url": url,
        "keyless": "" if os.environ.get(CUSTOM_ENV_KEY) else "1",
    }


def all_houses() -> Tuple[Dict[str, str], ...]:
    """The built-in houses plus the .env-configured one."""
    custom = _custom_spec()
    return HOUSES + ((custom,) if custom else ())


def env_number(name: str) -> Optional[float]:
    """A numeric setting from .env or the environment, or None."""
    load_dotenv()
    raw = (os.environ.get(name) or "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def house_spec(house_or_model: str) -> Dict[str, str]:
    """Look up a house by its id or model name, defaulting to DeepSeek."""
    token = (house_or_model or "").strip().lower()
    for spec in all_houses():
        if token in {spec["house"], spec["model"].lower(), spec["label"].lower()}:
            return spec
    return dict(HOUSES[0])


def available_houses() -> List[Dict[str, str]]:
    """Houses that can actually be called: a key is set, or none is needed."""
    load_dotenv()
    return [
        spec for spec in all_houses()
        if spec.get("keyless") or (spec["env"] and os.environ.get(spec["env"]))
    ]


def house_for_index(index: int) -> str:
    """Stable house assignment for a tribe or citizen id."""
    roster = available_houses()
    if not roster:
        return ""
    return roster[index % len(roster)]["house"]


@dataclass
class ChatResult:
    """Text from the model, or the reason there is none."""

    text: str = ""
    error: str = ""
    model: str = DEFAULT_MODEL
    status: int = 0  # HTTP status of the final attempt, 0 if none was made.
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_cached: int = 0  # Prompt tokens the provider served from its cache.

    @property
    def ok(self) -> bool:
        """True if the model returned usable text."""
        return bool(self.text) and not self.error

    @property
    def fatal(self) -> bool:
        """True when retrying cannot help: bad key, no access, or no such model."""
        return self.status in (401, 403, 404)


# Parameters some models refuse. When a 400 names one of these, the request is
# rewritten once (renamed or dropped) instead of failing for good -- providers
# change which knobs a model accepts more often than this file changes.
_RENAMES = {"max_tokens": "max_completion_tokens"}
_DROPPABLE = ("temperature", "reasoning_effort", "thinking", "response_format")


def _adapt_payload(payload: Dict[str, Any], detail: str) -> bool:
    """Rewrite a request a 400 complained about. True if anything changed."""
    lowered = detail.lower()
    for old, new in _RENAMES.items():
        if old in payload and old in lowered:
            payload[new] = payload.pop(old)
            return True
    for name in _DROPPABLE:
        if name in payload and name in lowered:
            del payload[name]
            return True
    return False


def _usage(body: Dict[str, Any]) -> Tuple[int, int, int]:
    """Prompt, completion and cached-prompt token counts, if reported."""
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(cached),
    )


def ssl_context() -> ssl.SSLContext:
    """A verifying SSL context that works on stock macOS Python installs.

    Python installed from python.org ships without a CA bundle until you run
    ``Install Certificates.command``, so the default context trusts nothing and
    every HTTPS call fails. When the system store is empty we fall back to the
    ``certifi`` bundle if it happens to be installed.

    Certificate verification is never disabled -- an unverified connection to a
    third party carrying an API key is not a trade worth making.
    """
    context = ssl.create_default_context()
    if not context.get_ca_certs():
        try:
            import certifi  # Optional; not a declared dependency.

            context.load_verify_locations(certifi.where())
        except (ImportError, OSError):
            pass
    return context


def is_available(explicit: Optional[str] = None) -> bool:
    """True if any model key is configured."""
    if explicit:
        return True
    return bool(available_houses())


def _openai_reasoning(model: str) -> bool:
    """GPT-5 and o-series models reason first and reject sampling knobs."""
    token = model.lower()
    return token.startswith("gpt-5") or (len(token) > 1 and token[0] == "o" and token[1].isdigit())


def chat(
    messages: List[Dict[str, str]],
    key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    timeout: float = 30.0,
    retries: int = 2,
    backoff: float = 1.5,
    temperature: float = 0.7,
    max_tokens: int = 1200,
    json_mode: bool = False,
    thinking: bool = False,
    reasoning_effort: Optional[str] = None,
) -> ChatResult:
    """Send a chat-completions request to DeepSeek or OpenAI.

    Thinking mode is off by default: tribe decisions have to come back in
    seconds or the live map stalls, and Flash without thinking is the whole
    point of using this model. JSON mode is for structured policies.

    Each house gets the parameters it understands: OpenAI reasoning models
    (GPT-5, o-series) take ``max_completion_tokens`` and no ``temperature``,
    and ``reasoning_effort`` keeps their hidden thinking -- which is billed --
    short. If a provider still rejects a parameter with a 400, the request is
    rewritten once and resent rather than failing every call for good.
    """
    spec = house_spec(model)
    resolved = key or (api_key(env_name=spec["env"]) if spec["env"] else None)
    if resolved is None and spec.get("keyless"):
        resolved = ""  # A local server, or the mock: no Authorization header.
    if resolved is None:
        # Fall back to whatever house is usable so a missing one does not
        # silence a tribe -- or the narrator -- that was assigned to it.
        roster = available_houses()
        if roster:
            spec = roster[0]
            resolved = (api_key(env_name=spec["env"]) if spec["env"] else None) or ""
            model = spec["model"]
    if spec["house"] == "mock":
        from .mock import reply

        return ChatResult(text=reply(messages), model=spec["model"], status=200)
    if resolved is None:
        return ChatResult(
            error=f"No API key. Put {ENV_KEY} or {OPENAI_ENV_KEY} in .env, or set LLM_BASE_URL.",
            model=model,
        )

    payload: Dict[str, Any] = {
        "model": spec["model"],
        "messages": messages,
        "stream": False,
    }
    if spec["house"] == "openai":
        payload["max_completion_tokens"] = max_tokens
        if _openai_reasoning(spec["model"]):
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
        else:
            payload["temperature"] = temperature
    else:
        payload["max_tokens"] = max_tokens
        payload["temperature"] = temperature
        if spec["house"] == "deepseek":
            payload["thinking"] = {"type": "enabled" if thinking else "disabled"}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error = ""
    last_status = 0
    body = None
    adapted = False
    attempt = 0
    while attempt <= retries:
        headers = {"Content-Type": "application/json"}
        if resolved:
            headers["Authorization"] = f"Bearer {resolved}"
        request = urllib.request.Request(
            spec["url"],
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=ssl_context()
            ) as response:
                last_status = response.status
                body = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as error:
            last_status = error.code
            detail = error.read().decode("utf-8", errors="ignore")[:300]
            if error.code == 400 and not adapted and _adapt_payload(payload, detail):
                adapted = True  # One free resend with the rewritten request.
                continue
            if error.code in RETRYABLE_STATUS and attempt < retries:
                last_error = f"API returned {error.code}"
                time.sleep(backoff * (2**attempt))
                attempt += 1
                continue
            if error.code in (401, 403):
                return ChatResult(
                    error=(
                        f"API rejected the key ({error.code}). Check that {spec['env']} "
                        f"is valid. {detail}"
                    ),
                    model=model,
                    status=error.code,
                )
            if error.code == 404:
                return ChatResult(
                    error=(
                        f"Model '{spec['model']}' is not available ({error.code}). "
                        "Check the model name in worldbox/ai/client.py:HOUSES."
                    ),
                    model=model,
                    status=error.code,
                )
            return ChatResult(
                error=f"API returned {error.code}: {detail}", model=model, status=error.code
            )
        except urllib.error.URLError as error:
            reason = error.reason
            if isinstance(reason, ssl.SSLError) or "CERTIFICATE_VERIFY_FAILED" in str(reason):
                return ChatResult(
                    error=(
                        f"TLS certificate verification failed ({reason}). Python installed "
                        "from python.org ships without a CA bundle -- run "
                        "'/Applications/Python 3.x/Install Certificates.command' once, "
                        "or 'pip install certifi'."
                    ),
                    model=model,
                )
            if attempt < retries:
                last_error = f"Could not reach the API: {reason}"
                time.sleep(backoff * (2**attempt))
                attempt += 1
                continue
            return ChatResult(error=f"Could not reach the API: {reason}", model=model)
        except (TimeoutError, OSError) as error:
            if attempt < retries:
                last_error = f"Network error: {error}"
                time.sleep(backoff * (2**attempt))
                attempt += 1
                continue
            return ChatResult(error=f"Network error: {error}", model=model)
        except json.JSONDecodeError:
            return ChatResult(error="The API returned a response that was not JSON.", model=model)

    if body is None:
        return ChatResult(error=last_error or "The request failed.", model=model, status=last_status)

    tokens_in, tokens_out, tokens_cached = _usage(body)
    try:
        message = body["choices"][0]["message"]
        text = (message.get("content") or "").strip()
    except (KeyError, IndexError, TypeError):
        return ChatResult(error="The API response contained no text.", model=model,
                          status=last_status, tokens_in=tokens_in, tokens_out=tokens_out)

    if not text:
        reason = ""
        try:
            reason = body["choices"][0].get("finish_reason", "")
        except (KeyError, IndexError, TypeError):
            pass
        if reason == "length":
            return ChatResult(
                error="The model hit its output limit before writing anything.",
                model=model, status=last_status, tokens_in=tokens_in, tokens_out=tokens_out,
            )
        return ChatResult(error="The model returned an empty reply.", model=model,
                          status=last_status, tokens_in=tokens_in, tokens_out=tokens_out)
    return ChatResult(
        text=text, model=model, status=last_status,
        tokens_in=tokens_in, tokens_out=tokens_out, tokens_cached=tokens_cached,
    )
