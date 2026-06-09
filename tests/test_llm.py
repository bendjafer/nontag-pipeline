import importlib
import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import nontag_pipeline.config as config


def _cache_key(system: str, prompt: str) -> str:
    content = json.dumps({"system": system, "prompt": prompt}, sort_keys=True)
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
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "  generated text  "}}]
    }
    mock_response.raise_for_status = MagicMock()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    with patch("requests.post", return_value=mock_response):
        result = llm.complete("prompt text", system="sys prompt")

    assert result == "generated text"
    key = _cache_key("sys prompt", "prompt text")
    cached = json.loads((tmp_path / f"{key}.json").read_text())
    assert cached["response"] == "generated text"


def test_missing_api_key_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import nontag_pipeline.llm as llm
    importlib.reload(llm)

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        llm.complete("hello", system="sys")
