import importlib
import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import nontag_pipeline.config as config


def _cache_key(system: str, prompt: str) -> str:
    content = json.dumps(
        {
            "system": system,
            "prompt": prompt,
            "backend": config.LLM_BACKEND,
            "model": config.LLM_MODEL,
            "base_url": config.LLM_BASE_URL,
            "temperature": config.LLM_TEMPERATURE,
            "seed": config.SEED,
        },
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()


def test_cache_hit_skips_api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    import nontag_pipeline.llm as llm
    importlib.reload(llm)

    key = _cache_key("sys", "hello")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"system": "sys", "prompt": "hello", "response": "cached!"})
    )

    with patch("requests.post") as mock_post:
        result = llm.complete("hello", system="sys")
        mock_post.assert_not_called()

    assert result == "cached!"


def test_cache_miss_calls_api_and_saves(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    import nontag_pipeline.llm as llm
    importlib.reload(llm)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "  generated text  "}}]
    }
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with patch("requests.post", return_value=mock_response):
        result = llm.complete("prompt text", system="sys prompt")

    assert result == "generated text"
    key = _cache_key("sys prompt", "prompt text")
    cached = json.loads((tmp_path / f"{key}.json").read_text())
    assert cached["response"] == "generated text"


def test_cache_key_depends_on_model(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    import nontag_pipeline.llm as llm
    importlib.reload(llm)

    key = _cache_key("sys", "hello")
    (tmp_path / f"{key}.json").write_text(
        json.dumps({"system": "sys", "prompt": "hello", "response": "old model output"})
    )

    monkeypatch.setattr(config, "LLM_MODEL", "some-other-model")
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "new model output"}}]
    }
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    with patch("requests.post", return_value=mock_response):
        result = llm.complete("hello", system="sys")

    assert result == "new model output"


def test_retry_on_429_then_success(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    import nontag_pipeline.llm as llm
    importlib.reload(llm)
    monkeypatch.setattr(llm, "_BACKOFF_BASE_S", 0.0)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    rate_limited = MagicMock()
    rate_limited.status_code = 429
    rate_limited.text = "rate limited"
    ok = MagicMock()
    ok.status_code = 200
    ok.json.return_value = {"choices": [{"message": {"content": "after retry"}}]}

    with patch("requests.post", side_effect=[rate_limited, ok]) as mock_post:
        result = llm.complete("retry prompt", system="sys")

    assert result == "after retry"
    assert mock_post.call_count == 2


def test_empty_response_raises_and_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    import nontag_pipeline.llm as llm
    importlib.reload(llm)
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"choices": [{"message": {"content": "   "}}]}

    with patch("requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="empty response"):
            llm.complete("hello", system="sys")

    assert list(tmp_path.iterdir()) == []


def test_missing_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    import nontag_pipeline.llm as llm
    importlib.reload(llm)

    with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
        llm.complete("hello", system="sys")
