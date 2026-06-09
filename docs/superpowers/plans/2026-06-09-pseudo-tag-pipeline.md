# Pseudo-TAG Textualization Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pipeline that takes a non-TAG graph (structure + labels, no text), generates styled synthetic text per node using PPR-selected neighbor signal, and saves the result as a pseudo-TAG graph compatible with LLM predictors (GraphGPT, LLaGA, GraphPrompter).

**Architecture:** Each file has one responsibility. `data.py` loads + splits the graph. `select.py` picks informative neighbors via PPR and computes label proportions. `narratives.py` maps class proportions to literary themes. `llm.py` calls OpenAI with on-disk caching. `textualize.py` ties selection → themes → generation into one call per node. `io.py` saves the pseudo-TAG. The demo script runs 5 nodes end-to-end and asserts no leakage.

**Tech Stack:** Python 3.12, PyTorch Geometric (Planetoid), NetworkX, PyTorch, requests (OpenAI HTTP), pytest. LLM: OpenAI (`OPENAI_API_KEY` env var). No new packages needed beyond what is in `.venv`.

---

## File Map

```
nontag_pipeline/
  __init__.py          empty package marker
  config.py            all tuneable constants — SEED, DATASET, STYLE, LLM_*, TARGET_LEN, paths
  data.py              load_dataset() + PUBMED_CLASSES + CORA_CLASSES constants
  select.py            ppr_selection() + neighbor_label_proportions()
  narratives.py        STYLE_TEMPLATES + LABEL_MAPS + map_proportions_to_themes()
  llm.py               complete(prompt, system) — OpenAI via requests + SHA-256 disk cache
  textualize.py        build_generation_prompt() + generate_node_text()
  io.py                save_pseudo_tag() — writes .pt (PyG) + .json

run_textualize_demo.py   smoke test: 5 test-mask nodes, prints prompt+output, asserts no leakage

tests/
  __init__.py
  test_select.py
  test_narratives.py
  test_textualize.py
  test_io.py

outputs/               created at runtime — pseudo_tag_pubmed_poetry.pt / .json
.cache/llm/            created at runtime — SHA-256 keyed JSON responses
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `nontag_pipeline/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create directories and empty init files**

```bash
mkdir -p nontag_pipeline tests
touch nontag_pipeline/__init__.py tests/__init__.py
```

- [ ] **Step 2: Verify structure**

```bash
find nontag_pipeline tests -type f | sort
```

Expected output:
```
nontag_pipeline/__init__.py
tests/__init__.py
```

- [ ] **Step 3: Commit**

```bash
git add nontag_pipeline/__init__.py tests/__init__.py
git commit -m "feat: scaffold nontag_pipeline package and tests directory"
```

---

## Task 2: config.py

**Files:**
- Create: `nontag_pipeline/config.py`

- [ ] **Step 1: Write config.py**

```python
# nontag_pipeline/config.py
SEED: int = 42
DATASET: str = "pubmed"        # "cora" | "pubmed"
STYLE: str = "poetry"

TARGET_LEN: int = 80           # target word count for generated text

PPR_K: int = 2                 # k-hop neighbourhood limit for PPR
PPR_M: int = 10                # top-m neighbours to select
PPR_ALPHA: float = 0.85

LLM_BACKEND: str = "openai"    # "openai" | "ollama"
LLM_MODEL: str = "gpt-4o-mini"
LLM_BASE_URL: str = "https://api.openai.com/v1"

CACHE_DIR: str = ".cache/llm"
OUTPUT_DIR: str = "outputs"
DATA_ROOT: str = "/tmp/planetoid"
```

- [ ] **Step 2: Verify import**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -c "from nontag_pipeline import config; print(config.DATASET)"
```

Expected: `pubmed`

- [ ] **Step 3: Commit**

```bash
git add nontag_pipeline/config.py
git commit -m "feat: add config.py with all tuneable constants"
```

---

## Task 3: data.py

**Files:**
- Create: `nontag_pipeline/data.py`

- [ ] **Step 1: Write data.py**

```python
# nontag_pipeline/data.py
from __future__ import annotations
import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid

PUBMED_CLASSES: list[str] = [
    "Diabetes_Mellitus_Experimental",
    "Diabetes_Mellitus_Type_1",
    "Diabetes_Mellitus_Type_2",
]

# NOTE: GraphGPT uses a 70-class Cora variant; swap load_dataset("cora") source later.
CORA_CLASSES: list[str] = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]

_CLASS_NAMES: dict[str, list[str]] = {
    "pubmed": PUBMED_CLASSES,
    "cora": CORA_CLASSES,
}


def load_dataset(
    name: str, seed: int = 42, root: str = "/tmp/planetoid"
) -> tuple[nx.Graph, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Load Planetoid dataset, apply 60/20/20 seeded split, return undirected graph."""
    dataset = Planetoid(root=root, name=name.capitalize())
    data = dataset[0]
    n: int = data.num_nodes

    torch.manual_seed(seed)
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    y = data.y.numpy()
    class_names = _CLASS_NAMES[name.lower()]
    return G, y, train_mask, val_mask, test_mask, class_names
```

- [ ] **Step 2: Smoke-test load**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -c "
from nontag_pipeline.data import load_dataset
G, y, tr, va, te, cls = load_dataset('pubmed')
print('nodes:', G.number_of_nodes(), 'edges:', G.number_of_edges())
print('train:', tr.sum().item(), 'val:', va.sum().item(), 'test:', te.sum().item())
print('classes:', cls)
"
```

Expected (PubMed has 19717 nodes):
```
nodes: 19717  edges: ~44338
train: 11830  val: 3943  test: 3944
classes: ['Diabetes_Mellitus_Experimental', 'Diabetes_Mellitus_Type_1', 'Diabetes_Mellitus_Type_2']
```

- [ ] **Step 3: Commit**

```bash
git add nontag_pipeline/data.py
git commit -m "feat: add data.py — load Planetoid with 60/20/20 seeded split"
```

---

## Task 4: select.py (TDD)

**Files:**
- Create: `nontag_pipeline/select.py`
- Create: `tests/test_select.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_select.py
import networkx as nx
import pytest
from nontag_pipeline.select import ppr_selection, neighbor_label_proportions


def _triangle() -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (0, 2)])
    return G


def test_ppr_excludes_self():
    G = _triangle()
    assert 0 not in ppr_selection(G, 0, k=2, m=10)


def test_ppr_returns_at_most_m():
    G = _triangle()
    assert len(ppr_selection(G, 0, k=2, m=2)) <= 2


def test_ppr_restrict_khop_limits_candidates():
    # star graph: centre=0, leaves=1..10
    G = nx.star_graph(10)
    # from leaf 1 with k=1: only 0 and siblings reachable in 1 hop (0 excluded, siblings via 0)
    # With k=1, only direct neighbors of node 1 are candidates: just node 0
    result = ppr_selection(G, 1, k=1, m=5, restrict_khop=True)
    # all results must be within 1 hop of node 1
    for u in result:
        assert nx.shortest_path_length(G, 1, u) <= 1


def test_proportions_all_visible():
    G = _triangle()
    visible = {1: 0, 2: 1}
    classes = ["ClassA", "ClassB"]
    props = neighbor_label_proportions(G, 0, [1, 2], visible, classes)
    assert abs(props["ClassA"] - 0.5) < 1e-9
    assert abs(props["ClassB"] - 0.5) < 1e-9


def test_proportions_skips_invisible_neighbors():
    G = _triangle()
    visible = {1: 0}          # node 2 is test — not in visible
    classes = ["ClassA", "ClassB"]
    props = neighbor_label_proportions(G, 0, [1, 2], visible, classes)
    assert props == {"ClassA": 1.0}


def test_proportions_no_visible_returns_empty():
    G = _triangle()
    props = neighbor_label_proportions(G, 0, [1, 2], {}, ["ClassA", "ClassB"])
    assert props == {}


def test_proportions_sum_to_one():
    G = nx.path_graph(5)
    visible = {1: 0, 2: 1, 3: 2}
    classes = ["A", "B", "C"]
    props = neighbor_label_proportions(G, 0, [1, 2, 3], visible, classes)
    assert abs(sum(props.values()) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_select.py -v 2>&1 | head -20
```

Expected: `ImportError` or `ModuleNotFoundError` — `select.py` does not exist yet.

- [ ] **Step 3: Write select.py**

```python
# nontag_pipeline/select.py
"""Neighbor Selection: Personalized PageRank (PPR) based selection."""
from __future__ import annotations
import networkx as nx


def ppr_selection(
    G: nx.Graph, v: int, k: int = 2, m: int = 10,
    alpha: float = 0.85, restrict_khop: bool = True
) -> list[int]:
    """Top-m nodes most relevant to v by Personalized PageRank."""
    ppr = nx.pagerank(G, alpha=alpha, personalization={v: 1.0})

    if restrict_khop:
        candidates = set(nx.single_source_shortest_path_length(G, v, cutoff=k))
    else:
        candidates = set(G.nodes())
    candidates.discard(v)

    ranked = sorted(candidates, key=lambda u: ppr.get(u, 0.0), reverse=True)
    return ranked[:m]


def neighbor_label_proportions(
    G: nx.Graph,
    v: int,
    selected: list[int],
    visible_labels: dict[int, int],
    class_names: list[str],
) -> dict[str, float]:
    """Proportion of each class among selected neighbors with visible labels."""
    counts: dict[str, int] = {}
    for u in selected:
        if u in visible_labels:
            name = class_names[visible_labels[u]]
            counts[name] = counts.get(name, 0) + 1

    total = sum(counts.values())
    if total == 0:
        return {}
    return {label: count / total for label, count in counts.items()}
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_select.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add nontag_pipeline/select.py tests/test_select.py
git commit -m "feat: add select.py — PPR neighbor selection + label proportions (TDD)"
```

---

## Task 5: narratives.py (TDD)

**Files:**
- Create: `nontag_pipeline/narratives.py`
- Create: `tests/test_narratives.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_narratives.py
from nontag_pipeline.data import PUBMED_CLASSES, CORA_CLASSES
from nontag_pipeline.narratives import LABEL_MAPS, map_proportions_to_themes


def test_label_map_covers_all_pubmed_classes():
    mapping = LABEL_MAPS[("pubmed", "poetry")]
    for cls in PUBMED_CLASSES:
        assert cls in mapping, f"Missing pubmed class in poetry map: {cls}"


def test_label_map_covers_all_cora_classes():
    mapping = LABEL_MAPS[("cora", "poetry")]
    for cls in CORA_CLASSES:
        assert cls in mapping, f"Missing cora class in poetry map: {cls}"


def test_map_proportions_sorted_descending():
    props = {"Diabetes_Mellitus_Type_1": 0.7, "Diabetes_Mellitus_Experimental": 0.3}
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert result[0][1] >= result[1][1]


def test_map_proportions_dominant_theme_first():
    props = {"Diabetes_Mellitus_Type_1": 0.7, "Diabetes_Mellitus_Experimental": 0.3}
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert result[0][0] == "the unbroken storm"
    assert result[1][0] == "fire and trial"


def test_map_proportions_returns_correct_length():
    props = {
        "Diabetes_Mellitus_Type_1": 0.5,
        "Diabetes_Mellitus_Experimental": 0.3,
        "Diabetes_Mellitus_Type_2": 0.2,
    }
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert len(result) == 3


def test_map_proportions_single_class():
    props = {"Diabetes_Mellitus_Type_2": 1.0}
    result = map_proportions_to_themes(props, "pubmed", "poetry")
    assert result == [("slow tide", 1.0)]
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_narratives.py -v 2>&1 | head -10
```

Expected: `ImportError` — `narratives.py` does not exist yet.

- [ ] **Step 3: Write narratives.py**

```python
# nontag_pipeline/narratives.py
"""Narrative style templates and class → theme mappings."""
from __future__ import annotations

STYLE_TEMPLATES: dict[str, str] = {
    "poetry": "a short lyric poem",
}

# Maps (dataset, style) -> {class_name: narrative_theme}
# Themes evoke class content without naming it — edit freely to improve generation quality.
LABEL_MAPS: dict[tuple[str, str], dict[str, str]] = {
    ("pubmed", "poetry"): {
        "Diabetes_Mellitus_Experimental": "fire and trial",
        "Diabetes_Mellitus_Type_1": "the unbroken storm",
        "Diabetes_Mellitus_Type_2": "slow tide",
    },
    ("cora", "poetry"): {
        "Case_Based": "keeper of stories",
        "Genetic_Algorithms": "seeds and seasons",
        "Neural_Networks": "the weaving mind",
        "Probabilistic_Methods": "fog and chance",
        "Reinforcement_Learning": "the returning traveler",
        "Rule_Learning": "law and stone",
        "Theory": "the open horizon",
    },
}


def map_proportions_to_themes(
    proportions: dict[str, float], dataset: str, style: str
) -> list[tuple[str, float]]:
    """Map class proportions to (theme, pct) pairs, sorted descending by proportion."""
    label_map = LABEL_MAPS[(dataset, style)]
    themed = [(label_map[cls], pct) for cls, pct in proportions.items() if cls in label_map]
    return sorted(themed, key=lambda x: x[1], reverse=True)
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_narratives.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add nontag_pipeline/narratives.py tests/test_narratives.py
git commit -m "feat: add narratives.py — style templates + poetry label maps for cora/pubmed (TDD)"
```

---

## Task 6: llm.py (TDD)

**Files:**
- Create: `nontag_pipeline/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_llm.py
import json
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

import nontag_pipeline.config as config


def _cache_key(system: str, prompt: str) -> str:
    content = json.dumps({"system": system, "prompt": prompt}, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def test_cache_hit_skips_api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE_DIR", str(tmp_path))
    from nontag_pipeline import llm
    importlib_reload(llm)

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
    from nontag_pipeline import llm
    importlib_reload(llm)

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
    from nontag_pipeline import llm
    importlib_reload(llm)

    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        llm.complete("hello", system="sys")


def importlib_reload(module):
    import importlib
    importlib.reload(module)
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_llm.py -v 2>&1 | head -10
```

Expected: `ImportError` — `llm.py` does not exist yet.

- [ ] **Step 3: Write llm.py**

```python
# nontag_pipeline/llm.py
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
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_llm.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add nontag_pipeline/llm.py tests/test_llm.py
git commit -m "feat: add llm.py — OpenAI/Ollama backend with SHA-256 disk cache (TDD)"
```

---

## Task 7: textualize.py (TDD)

**Files:**
- Create: `nontag_pipeline/textualize.py`
- Create: `tests/test_textualize.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_textualize.py
import networkx as nx
import torch
import numpy as np
from unittest.mock import patch
from nontag_pipeline.textualize import build_generation_prompt, generate_node_text


def test_prompt_system_forbids_graph_terms():
    system, _ = build_generation_prompt([("the unbroken storm", 0.7)], "poetry", 80)
    forbidden = ["graph", "node", "neighbor", "label", "categor", "percentage"]
    lowered = system.lower()
    for word in forbidden:
        assert word in lowered, f"System prompt must mention forbidding '{word}'"


def test_prompt_user_contains_all_themes():
    themes = [("the unbroken storm", 0.7), ("fire and trial", 0.3)]
    _, user = build_generation_prompt(themes, "poetry", 80)
    assert "the unbroken storm" in user
    assert "fire and trial" in user


def test_prompt_user_contains_target_len():
    _, user = build_generation_prompt([("slow tide", 1.0)], "poetry", 120)
    assert "120" in user


def test_prompt_user_contains_style():
    _, user = build_generation_prompt([("fog and chance", 1.0)], "poetry", 80)
    assert "poetry" in user


def test_prompt_percentages_shown_as_integers():
    _, user = build_generation_prompt([("slow tide", 0.666)], "poetry", 80)
    assert "67%" in user


def _make_small_graph():
    """3-node graph: node 0 (test), nodes 1+2 (train)."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    y = np.array([1, 0, 1])                  # labels
    train_mask = torch.tensor([False, True, True])
    val_mask   = torch.tensor([False, False, False])
    test_mask  = torch.tensor([True,  False, False])
    class_names = ["Diabetes_Mellitus_Experimental", "Diabetes_Mellitus_Type_1"]
    return G, y, train_mask, val_mask, test_mask, class_names


def test_generate_returns_ok_when_neighbors_visible():
    G, y, tr, va, te, cls = _make_small_graph()
    visible = {n: int(y[n]) for n in range(len(y)) if tr[n] or va[n]}

    with patch("nontag_pipeline.llm.complete", return_value="a poem"):
        result = generate_node_text(
            G, 0, visible, cls, "pubmed", "poetry", 80
        )
    assert result["status"] == "ok"
    assert result["text"] == "a poem"
    assert result["node"] == 0


def test_generate_returns_no_signal_when_no_visible_neighbors():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (0, 2)])
    y = np.array([1, 0, 1])
    class_names = ["ClassA", "ClassB"]
    visible = {}   # no train/val labels at all

    result = generate_node_text(G, 0, visible, class_names, "pubmed", "poetry", 80)
    assert result["status"] == "no_signal"
    assert result["text"] is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_textualize.py -v 2>&1 | head -10
```

Expected: `ImportError` — `textualize.py` does not exist yet.

- [ ] **Step 3: Write textualize.py**

```python
# nontag_pipeline/textualize.py
"""Prompt construction and per-node text generation."""
from __future__ import annotations
import numpy as np
import networkx as nx

from nontag_pipeline import llm
from nontag_pipeline.narratives import map_proportions_to_themes
from nontag_pipeline.select import neighbor_label_proportions, ppr_selection
from nontag_pipeline import config

_SYSTEM = (
    "You generate a single synthetic text in a specified literary style. Never mention graphs, "
    "nodes, neighbors, labels, categories, or percentages. Output only the text."
)


def build_generation_prompt(
    themes_with_pct: list[tuple[str, float]], style: str, target_len: int
) -> tuple[str, str]:
    """Return (system, user) prompt pair for the given themes and style."""
    theme_lines = "\n".join(
        f"- {theme}: {round(pct * 100)}%" for theme, pct in themes_with_pct
    )
    user = (
        "This item belongs to a homophilic network, so it most likely shares the themes of the "
        "important related items selected around it.\n"
        f"Themes and their proportions:\n{theme_lines}\n"
        f"Write one {style} of about {target_len} words whose content embodies these themes "
        "in those proportions: the dominant theme sets the overall mood, the minor themes are "
        "woven in.\n"
        "Do NOT name the themes or state any numbers — evoke them only through imagery and content."
    )
    return _SYSTEM, user


def generate_node_text(
    G: nx.Graph,
    v: int,
    visible_labels: dict[int, int],
    class_names: list[str],
    dataset: str,
    style: str,
    target_len: int,
    k: int = config.PPR_K,
    m: int = config.PPR_M,
) -> dict:
    """Select neighbors, compute proportions, generate styled text for node v."""
    selected = ppr_selection(G, v, k=k, m=m)
    proportions = neighbor_label_proportions(G, v, selected, visible_labels, class_names)

    if not proportions:
        return {"node": v, "status": "no_signal", "text": None}

    themes = map_proportions_to_themes(proportions, dataset, style)
    system, user = build_generation_prompt(themes, style, target_len)
    text = llm.complete(user, system=system)
    return {"node": v, "status": "ok", "text": text, "themes": themes}
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_textualize.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add nontag_pipeline/textualize.py tests/test_textualize.py
git commit -m "feat: add textualize.py — prompt builder + node text generation (TDD)"
```

---

## Task 8: io.py (TDD)

**Files:**
- Create: `nontag_pipeline/io.py`
- Create: `tests/test_io.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_io.py
import json
import numpy as np
import networkx as nx
import torch
import pytest
from pathlib import Path
from nontag_pipeline.io import save_pseudo_tag


def _minimal_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])
    y = np.array([0, 1, 2])
    train_mask = torch.tensor([True,  False, False])
    val_mask   = torch.tensor([False, True,  False])
    test_mask  = torch.tensor([False, False, True])
    class_names = ["ClassA", "ClassB", "ClassC"]
    records = [
        {"node": 0, "status": "ok",        "text": "poem for 0"},
        {"node": 1, "status": "ok",        "text": "poem for 1"},
        {"node": 2, "status": "no_signal", "text": None},
    ]
    return G, y, train_mask, val_mask, test_mask, class_names, records


def test_save_creates_pt_file(tmp_path):
    args = _minimal_graph()
    pt_path, _ = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    assert pt_path.exists()


def test_save_creates_json_file(tmp_path):
    args = _minimal_graph()
    _, json_path = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    assert json_path.exists()


def test_json_contains_all_nodes(tmp_path):
    args = _minimal_graph()
    _, json_path = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    data = json.loads(json_path.read_text())
    assert len(data["nodes"]) == 3


def test_json_splits_are_correct(tmp_path):
    args = _minimal_graph()
    _, json_path = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    data = json.loads(json_path.read_text())
    splits = {n["node_id"]: n["split"] for n in data["nodes"]}
    assert splits[0] == "train"
    assert splits[1] == "val"
    assert splits[2] == "test"


def test_pt_has_raw_texts(tmp_path):
    args = _minimal_graph()
    pt_path, _ = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    saved = torch.load(pt_path, weights_only=False)
    assert saved.raw_texts[0] == "poem for 0"
    assert saved.raw_texts[2] is None


def test_pt_edge_index_shape(tmp_path):
    args = _minimal_graph()
    pt_path, _ = save_pseudo_tag(*args, output_dir=tmp_path, dataset="pubmed", style="poetry")
    saved = torch.load(pt_path, weights_only=False)
    assert saved.edge_index.shape[0] == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_io.py -v 2>&1 | head -10
```

Expected: `ImportError` — `io.py` does not exist yet.

- [ ] **Step 3: Write io.py**

```python
# nontag_pipeline/io.py
"""Save pseudo-TAG graph as PyG .pt and human-readable .json."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data


def save_pseudo_tag(
    G: nx.Graph,
    y: np.ndarray,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    test_mask: torch.Tensor,
    class_names: list[str],
    records: list[dict],
    output_dir: str | Path,
    dataset: str,
    style: str,
) -> tuple[Path, Path]:
    """Write pseudo-TAG to <output_dir>/pseudo_tag_<dataset>_<style>.{pt,json}."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"pseudo_tag_{dataset}_{style}"

    raw_texts: list[str | None] = [None] * G.number_of_nodes()
    for rec in records:
        raw_texts[rec["node"]] = rec.get("text")

    edges = list(G.edges())
    if edges:
        src = [u for u, _ in edges] + [v for _, v in edges]
        dst = [v for _, v in edges] + [u for u, _ in edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    pyg_data = Data(
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.long),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    pyg_data.raw_texts = raw_texts
    pyg_data.class_names = class_names

    pt_path = out / f"{stem}.pt"
    torch.save(pyg_data, pt_path)

    def _split(n: int) -> str:
        if train_mask[n].item():
            return "train"
        if val_mask[n].item():
            return "val"
        return "test"

    json_obj = {
        "dataset": dataset,
        "style": style,
        "class_names": class_names,
        "nodes": [
            {
                "node_id": int(n),
                "text": raw_texts[n],
                "label": int(y[n]),
                "label_name": class_names[int(y[n])],
                "split": _split(n),
            }
            for n in G.nodes()
        ],
        "edges": [[int(u), int(v)] for u, v in G.edges()],
    }
    json_path = out / f"{stem}.json"
    json_path.write_text(json.dumps(json_obj, indent=2))

    return pt_path, json_path
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/test_io.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add nontag_pipeline/io.py tests/test_io.py
git commit -m "feat: add io.py — save pseudo-TAG as PyG .pt + .json (TDD)"
```

---

## Task 9: run_textualize_demo.py (integration smoke test)

**Files:**
- Create: `run_textualize_demo.py`

- [ ] **Step 1: Write run_textualize_demo.py**

```python
# run_textualize_demo.py
"""Smoke test: textualize 5 test-mask nodes, assert no leakage, save pseudo-TAG."""
import random
from nontag_pipeline import config
from nontag_pipeline.data import load_dataset
from nontag_pipeline.textualize import generate_node_text
from nontag_pipeline.io import save_pseudo_tag


def main() -> None:
    G, y, train_mask, val_mask, test_mask, class_names = load_dataset(
        config.DATASET, seed=config.SEED, root=config.DATA_ROOT
    )

    # Build visible_labels ONCE from train + val only
    visible_labels = {
        int(n): int(y[n])
        for n in G.nodes()
        if train_mask[n].item() or val_mask[n].item()
    }

    # Assert: no test node label is present in visible_labels (leakage check)
    test_nodes = [int(n) for n in G.nodes() if test_mask[n].item()]
    for n in test_nodes:
        assert n not in visible_labels, f"Leakage: test node {n} found in visible_labels"

    # Pick 5 test nodes deterministically
    rng = random.Random(config.SEED)
    sample = rng.sample(test_nodes, 5)

    records: list[dict] = []
    for node_id in sample:
        print(f"\n{'=' * 60}")
        print(f"Node {node_id}  (test, true label: {class_names[int(y[node_id])]})")

        result = generate_node_text(
            G, node_id, visible_labels, class_names,
            config.DATASET, config.STYLE, config.TARGET_LEN,
        )
        records.append(result)

        print(f"Status : {result['status']}")
        if result["status"] == "ok":
            print(f"Themes : {result['themes']}")
            print(f"\nGenerated text:\n{result['text']}")
        else:
            print("No visible-labeled neighbors — skipped LLM call.")

    print(f"\n{'=' * 60}")
    print(f"Leakage assertion passed for all {len(test_nodes)} test nodes.")
    print("Demo complete. Full pipeline run and save are separate steps.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Set OPENAI_API_KEY and run**

```bash
export OPENAI_API_KEY="your-key-here"
cd /home/qamar/sandbox2 && .venv/bin/python run_textualize_demo.py
```

Expected output (5 nodes):
```
============================================================
Node 4521  (test, true label: Diabetes_Mellitus_Type_1)
Status : ok
Themes : [('the unbroken storm', 0.75), ('fire and trial', 0.25)]

Generated text:
<poem from OpenAI>
...
Leakage assertion passed for all 3944 test nodes.
Demo complete.
```

- [ ] **Step 3: Run full test suite to confirm nothing regressed**

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add run_textualize_demo.py
git commit -m "feat: add run_textualize_demo.py — 5-node smoke test with leakage assertion"
```

---

## Full Test Suite Summary

```bash
cd /home/qamar/sandbox2 && .venv/bin/python -m pytest tests/ -v --tb=short
```

Expected:
```
tests/test_select.py     7 passed
tests/test_narratives.py 6 passed
tests/test_llm.py        3 passed
tests/test_textualize.py 7 passed
tests/test_io.py         6 passed
```

---

## Notes for Later Steps (do NOT implement now)

- Step 4 (classify): iterate all nodes, call `generate_node_text`, collect records, call `save_pseudo_tag`
- Step 5 (export): add `export_to_graphgpt.py`, `export_to_llaga.py` once those repos are cloned
- Cora swap: change `config.DATASET = "cora"` — no other changes needed
- University LLM: change `config.LLM_BACKEND`, `config.LLM_BASE_URL`, `config.LLM_MODEL` — no code changes
