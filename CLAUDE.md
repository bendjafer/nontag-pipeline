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
- **Narrative remapping**: original class names are mapped to themes in a chosen style (e.g.
  poetry), so generated text evokes themes, not literal class names.

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
  narratives.py        STYLE_TEMPLATES + LABEL_MAPS + map_proportions_to_themes()
  llm.py               complete() — OpenAI/Ollama via requests + SHA-256 disk cache
  textualize.py        build_generation_prompt() + generate_node_text()
  io.py                save_pseudo_tag() — writes .pt (PyG Data) + .json

run_textualize_demo.py   smoke test: 5 test nodes, prints prompt+output, asserts no leakage

tests/
  test_select.py       (9 tests)
  test_narratives.py   (6 tests)
  test_llm.py          (3 tests)
  test_textualize.py   (7 tests)
  test_io.py           (6 tests)

outputs/               created at runtime — pseudo_tag_pubmed_poetry.pt / .json
.cache/llm/            created at runtime — SHA-256 keyed JSON LLM response cache
docs/superpowers/      implementation plan + design spec
```

### How to Run

```bash
# Install dependencies (already in .venv)
# .venv has: torch, torch_geometric, networkx, requests, pytest

# Run tests (no API key needed)
.venv/bin/python -m pytest tests/ -v

# Run smoke test (requires OpenAI API key)
export LLM_API_KEY="sk-..."        # OpenAI key, university key, or whatever your server uses
.venv/bin/python run_textualize_demo.py

# To switch to Cora: edit nontag_pipeline/config.py → DATASET = "cora"
# To switch LLM to Ollama: edit config.py → LLM_BACKEND = "ollama", LLM_BASE_URL = "http://localhost:11434"
# To switch to university server: edit config.py → LLM_BACKEND = "openai", LLM_BASE_URL = "<server_url>"
```

### LLM Backend
- Default: OpenAI (`gpt-4o-mini`), reads `LLM_API_KEY` from environment — never hardcoded.
- Ollama supported: change `LLM_BACKEND = "ollama"` and `LLM_BASE_URL = "http://localhost:11434"`.
- University server: change `LLM_BASE_URL` to the server endpoint — zero code changes.
- All responses cached on disk at `.cache/llm/<sha256>.json` — re-running never re-calls the API.

### Saved Output Format
`save_pseudo_tag()` writes two files to `outputs/`:
- **`pseudo_tag_<dataset>_<style>.pt`** — PyG `Data` object with:
  - `data.edge_index`, `data.y`, `data.train_mask`, `data.val_mask`, `data.test_mask`
  - `data.raw_texts` — list of str|None, one per node (None = no_signal nodes)
  - `data.class_names` — list of class name strings
- **`pseudo_tag_<dataset>_<style>.json`** — human-readable:
  - `nodes`: list of `{node_id, text, label, label_name, split}`
  - `edges`: list of `[u, v]` pairs
  - `dataset`, `style`, `class_names`

The `.pt` format is compatible with GraphGPT, LLaGA, and GraphPrompter (all use `raw_texts`).
When those repos are cloned, add a thin `export_to_<model>.py` adapter if needed.

---

## Coding Conventions
- One responsibility per file; short, single-purpose functions; clear, consistent names.
- Type hints + one-line docstrings. No speculative features, no premature abstraction.
- Config-driven (dataset, style, params in config.py) so the same code runs on any dataset.
- No hardcoded secrets. Deterministic where possible (seed everything).
- Tests live in `tests/` — run with `pytest`.

## Narrative Mappings (Poetry Style)

### PubMed (3 classes)
| Class | Poetry Theme |
|---|---|
| Diabetes_Mellitus_Experimental | fire and trial |
| Diabetes_Mellitus_Type_1 | the unbroken storm |
| Diabetes_Mellitus_Type_2 | slow tide |

### Cora (7 classes)
| Class | Poetry Theme |
|---|---|
| Case_Based | keeper of stories |
| Genetic_Algorithms | seeds and seasons |
| Neural_Networks | the weaving mind |
| Probabilistic_Methods | fog and chance |
| Reinforcement_Learning | the returning traveler |
| Rule_Learning | law and stone |
| Theory | the open horizon |

To add a new style: add an entry to `STYLE_TEMPLATES` and a new key to `LABEL_MAPS` in
`nontag_pipeline/narratives.py`, then change `config.STYLE`.

## Known Notes for Next Steps
- **Cora GraphGPT variant**: GraphGPT uses a 70-class Cora. The current `CORA_CLASSES` is the
  standard 7-class Planetoid Cora. Swap the data source in `data.py` when moving to GraphGPT Cora.
- **Full graph textualization**: `run_textualize_demo.py` only runs 5 nodes. To textualize all
  nodes: iterate `G.nodes()`, call `generate_node_text()` for each, collect records, call
  `save_pseudo_tag()`. Add this as Step 4 script.
- **Ollama base URL**: When `LLM_BACKEND = "ollama"`, also set `LLM_BASE_URL = "http://localhost:11434"`.
  The default URL is the OpenAI endpoint and will not work for Ollama.
