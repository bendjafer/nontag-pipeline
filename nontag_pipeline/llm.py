"""LLM backend with on-disk SHA-256 cache. Backend swappable via config."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

import requests

from nontag_pipeline import config


def complete(prompt: str, system: str | None = None) -> str:
    """Call configured LLM backend; return cached response if available."""
    system = system or "You are a helpful assistant."
    key = _cache_key(system, prompt)
    path = _cache_path(key)

    if path.exists():
        return json.loads(path.read_text())["response"]

    if config.LLM_BACKEND == "openai":
        response = _call_openai(system, prompt)
    elif config.LLM_BACKEND == "ollama":
        response = _call_ollama(system, prompt)
    else:
        raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}")

    path.write_text(json.dumps({"system": system, "prompt": prompt, "response": response}))
    return response


def _cache_key(system: str, prompt: str) -> str:
    content = json.dumps({"system": system, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def _cache_path(key: str) -> Path:
    cache_dir = Path(config.CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _call_openai(system: str, prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set")
    resp = requests.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _call_ollama(system: str, prompt: str) -> str:
    resp = requests.post(
        f"{config.LLM_BASE_URL}/api/chat",
        json={
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()
