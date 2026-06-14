"""LLM backend with on-disk SHA-256 cache. Backend swappable via config."""
from __future__ import annotations
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

import requests

_MAX_ATTEMPTS = 5
_BACKOFF_BASE_S = 1.0  # delays: 1, 2, 4, 8, 16 seconds between attempts

from nontag_pipeline import config


def complete(prompt: str, system: str | None = None, seed: int | None = None) -> str:
    """Call configured LLM backend; return cached response if available.

    Pass seed=node_id to guarantee unique outputs for nodes that share the
    same proportion profile. The seed is included in the cache key so each
    (prompt, seed) pair is cached independently.
    """
    system = system or "You are a helpful assistant."
    effective_seed = seed if seed is not None else config.SEED
    key = _cache_key(system, prompt, effective_seed)
    path = _cache_path(key)

    if path.exists():
        return json.loads(path.read_text())["response"]

    if config.LLM_BACKEND == "openai":
        response = _call_openai(system, prompt, effective_seed)
    elif config.LLM_BACKEND == "ollama":
        response = _call_ollama(system, prompt, effective_seed)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}")

    if not response:
        raise RuntimeError("LLM returned an empty response; not caching it")

    # Atomic write: unique tmp name per process+call avoids collisions when
    # two processes cache the same key concurrently.
    tmp_path = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.tmp")
    tmp_path.write_text(json.dumps({"system": system, "prompt": prompt, "response": response}))
    os.replace(tmp_path, path)
    return response


def _cache_key(system: str, prompt: str, seed: int) -> str:
    content = json.dumps(
        {
            "system": system,
            "prompt": prompt,
            "backend": config.LLM_BACKEND,
            "model": config.LLM_MODEL,
            "base_url": config.LLM_BASE_URL,
            "temperature": config.LLM_TEMPERATURE,
            "seed": seed,
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    cache_dir = Path(config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _post_with_retry(url: str, *, headers: dict | None = None, json_body: dict,
                     timeout: int) -> requests.Response:
    """POST with exponential backoff on 429/5xx and connection errors."""
    last_error: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = requests.post(url, headers=headers, json=json_body, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            else:
                resp.raise_for_status()
                return resp
        except requests.ConnectionError as e:
            last_error = e
        except requests.Timeout as e:
            last_error = e
        time.sleep(_BACKOFF_BASE_S * 2 ** attempt)
    raise RuntimeError(f"LLM request failed after {_MAX_ATTEMPTS} attempts: {last_error}")


def _call_openai(system: str, prompt: str, seed: int) -> str:
    api_key = os.environ.get(config.LLM_KEY_ENV)
    if not api_key:
        raise EnvironmentError(f"{config.LLM_KEY_ENV} is not set")
    resp = _post_with_retry(
        f"{config.LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json_body={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": config.LLM_TEMPERATURE,
            "seed": seed,
        },
        timeout=60,
    )
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(system: str, prompt: str, seed: int) -> str:
    resp = _post_with_retry(
        f"{config.LLM_BASE_URL}/api/chat",
        json_body={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": config.LLM_TEMPERATURE, "seed": seed},
        },
        timeout=120,
    )
    return resp.json()["message"]["content"].strip()
