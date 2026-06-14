# Project: non-TAG → pseudo-TAG Textualization Pipeline

## Goal
Take a graph with NO node text (non-TAG = structure + labels), generate synthetic node text
("pseudo-TAG") so a TAG-based LLM predictor (e.g. GraphGPT, LLaGA, GraphPrompter) can
classify it. We study whether the NARRATIVE STYLE of the generated text affects accuracy.
Datasets: PubMed (3 classes) now, Cora (7 classes) later. Graphs are homophilic (neighbors
usually share a class).

## Key Definitions
- **non-TAG**: graph with edges + labels but no node text.
- **pseudo-TAG**: the non-TAG after we attach generated text to each node.
- **Homophily**: connected nodes tend to share the same class → neighbor labels are informative.
- **Narrative remapping**: original class names are mapped to fixed *topics* (war / love /
  nature …), so generated text evokes the topic, not the literal class name. The topic
  assignment is fixed per dataset; the *style* (poetry / news / story) only changes how the
  topic is rendered — so style comparisons are controlled by construction.

## THE LEAKAGE RULE (non-negotiable)
- A node's generated text may use: graph structure + labels of neighbors in train_mask | val_mask.
- It must NEVER use: that node's own label, or any test_mask label.
- Test labels are used only for final scoring (later steps). Enforce with assertions.
- How it's enforced: `visible_labels = {n: y[n] for n in G.nodes() if train_mask[n] or val_mask[n]}`
  is built ONCE and is the ONLY label source passed to generation. Test nodes are absent from
  this dict by construction — no explicit filtering needed downstream.

## Generation Principle
- Condition the text on the proportion of selected-neighbor themes (e.g. 70/20/10).
- EVOKE themes through content; do NOT name themes or state percentages in the output.
  (Naming them would turn later classification into trivial keyword matching.)

## Why Separation of Textualization and Prediction is Correct
- Textualize ALL nodes (train, val, test) once → save pseudo-TAG.
- Feed to any LLM predictor (GraphGPT, LLaGA, GraphPrompter) as input.
- Test nodes get text generated from their train/val neighbors' signal only — no own label used.
- The predictor does real inference from text + structure; it doesn't read back the answer.

## Pipeline Stages
1. **Data** [DONE]: load + 60/20/20 seeded split + strip text → undirected networkx graph.
2. **Select** [DONE]: Personalized PageRank top-m neighbors of a node.
3. **Textualize** [DONE]: proportions → narrative themes → LLM generates styled text.
4. **Classify** [TODO]: run LLM predictor (GraphGPT / LLaGA / GraphPrompter) on pseudo-TAG.
5. **Baselines** [TODO]: label propagation, GCN.
6. **Evaluation** [TODO]: accuracy comparison across narrative styles.

---

## Implementation Status: Steps 1–3 Complete

### File Map

```
nontag_pipeline/
  __init__.py          empty package marker
  config.py            all tuneable constants — change DATASET/STYLE here, nothing else changes
  data.py              load_dataset() → (G, y, train_mask, val_mask, test_mask, class_names)
  select.py            ppr_selection() + neighbor_label_proportions()
  narratives.py        STYLE_TEMPLATES (rendering) + TOPIC_MAPS (fixed topics) + map_proportions_to_themes()
  llm.py               complete() — OpenAI/Ollama via requests + SHA-256 disk cache
  textualize.py        build_generation_prompt() + generate_node_text()
  io.py                save_pseudo_tag() — writes .pt (PyG Data) + .json

run_textualize_demo.py   smoke test: 5 test nodes, prints prompt+output (saves nothing)
run_textualize_full.py   Step 4 driver: textualize ALL nodes, save pseudo-TAG to outputs/

tests/
  test_data.py         (5 tests)
  test_select.py       (9 tests)
  test_narratives.py   (6 tests)
  test_llm.py          (6 tests)
  test_textualize.py   (9 tests)
  test_io.py           (7 tests)

outputs/               created by run_textualize_full.py — pseudo_tag_pubmed_poetry.pt / .json
data/                  created at runtime — Planetoid download cache (gitignored)
.cache/llm/            created at runtime — SHA-256 keyed JSON LLM response cache
docs/superpowers/      implementation plan + design spec
```

### How to Run

```bash
# Install dependencies (already in .venv)
# .venv has: torch, torch_geometric, networkx, requests, pytest

# Run tests (no API key needed)
.venv/bin/python -m pytest tests/ -v

# Run smoke test (requires OpenAI API key; prints 5 nodes, saves nothing)
export LLM_API_KEY="sk-..."        # OpenAI key, university key, or whatever your server uses
.venv/bin/python run_textualize_demo.py

# Full run: textualize ALL nodes and save the pseudo-TAG to outputs/
.venv/bin/python run_textualize_full.py

# To switch to Cora: edit nontag_pipeline/config.py → DATASET = "cora"
# To switch LLM to Ollama: edit config.py → LLM_BACKEND = "ollama", LLM_BASE_URL = "http://localhost:11434"
# To switch to university server: edit config.py → LLM_BACKEND = "openai", LLM_BASE_URL = "<server_url>"
```

### LLM Backend
- Default: OpenAI (`gpt-4o-mini`), reads `LLM_API_KEY` from environment — never hardcoded.
- Ollama supported: change `LLM_BACKEND = "ollama"` and `LLM_BASE_URL = "http://localhost:11434"`.
- University server: change `LLM_BASE_URL` to the server endpoint — zero code changes.
- Deterministic: `temperature = 0` (config.LLM_TEMPERATURE) + `seed = config.SEED` sent with
  every request (OpenAI seed is best-effort).
- Transient failures (429, 5xx, connection errors) retried 5x with exponential backoff.
- All responses cached on disk at `.cache/llm/<sha256>.json` — re-running never re-calls the API.
  Cache key includes backend, model, base URL, temperature, and seed, so switching any of
  these never reuses stale responses. Empty responses are rejected, never cached.

### Saved Output Format
`save_pseudo_tag()` writes two files to `outputs/`:
- **`pseudo_tag_<dataset>_<style>.pt`** — PyG `Data` object with:
  - `data.edge_index`, `data.y`, `data.train_mask`, `data.val_mask`, `data.test_mask`
  - `data.raw_texts` — list of str|None, one per node (None = no_signal nodes)
  - `data.class_names` — list of class name strings
- **`pseudo_tag_<dataset>_<style>.json`** — human-readable, for inspection/scoring ONLY
  (contains true labels for ALL nodes incl. test — never feed it to a predictor):
  - `nodes`: list of `{node_id, text, label, label_name, split, n_selected, n_visible}`
    (`n_visible` of `n_selected` PPR neighbors had train/val labels — evidence strength)
  - `edges`: list of `[u, v]` pairs
  - `dataset`, `style`, `class_names`, `warning`

The `.pt` format is compatible with GraphGPT, LLaGA, and GraphPrompter (all use `raw_texts`).
When those repos are cloned, add a thin `export_to_<model>.py` adapter if needed.

---

## Coding Conventions
- One responsibility per file; short, single-purpose functions; clear, consistent names.
- Type hints + one-line docstrings. No speculative features, no premature abstraction.
- Config-driven (dataset, style, params in config.py) so the same code runs on any dataset.
- No hardcoded secrets. Deterministic where possible (seed everything).
- Tests live in `tests/` — run with `pytest`.

## Topic Mappings (fixed per dataset, shared by every style)

The class → topic assignment is in `TOPIC_MAPS` (`narratives.py`). It does NOT depend on style:
every style renders the *same* topics, so accuracy differences across styles are attributable to
style alone. Topics must be mutually distinct (separability drives accuracy) and are never named
in the output (the prompt forces evoke-don't-name, avoiding trivial keyword matching).

### PubMed (3 classes)
| Class | Topic |
|---|---|
| Diabetes_Mellitus_Experimental | the laboratory |
| Diabetes_Mellitus_Type_1 | the siege |
| Diabetes_Mellitus_Type_2 | the harvest |

### Cora (7 classes)
| Class | Topic |
|---|---|
| Case_Based | the archive |
| Genetic_Algorithms | evolution |
| Neural_Networks | the mind |
| Probabilistic_Methods | the fog |
| Reinforcement_Learning | the arena |
| Rule_Learning | the ledger |
| Theory | the cosmos |

### Styles (rendering only)
`STYLE_TEMPLATES` in `narratives.py`: `poetry` → "a short lyric poem", `news` → "a short news
report", `story` → "a brief short story".

- **To compare styles** (the research question): set `config.STYLE` to `poetry` / `news` /
  `story` and run `run_textualize.py` for each. Topics stay fixed; only the rendering
  changes. Each run writes `outputs/pseudo_tag_<dataset>_<style>.{pt,json}`.
- **To add a style**: add one entry to `STYLE_TEMPLATES`, then set `config.STYLE`.
- **To study topic distinctness** (a different axis): edit the topic values in `TOPIC_MAPS`.

## Known Notes for Next Steps
- **Cora GraphGPT variant**: GraphGPT uses a 70-class Cora. The current `CORA_CLASSES` is the
  standard 7-class Planetoid Cora. Swap the data source in `data.py` when moving to GraphGPT Cora.
- **Ollama base URL**: When `LLM_BACKEND = "ollama"`, also set `LLM_BASE_URL = "http://localhost:11434"`.
  The default URL is the OpenAI endpoint and will not work for Ollama.
