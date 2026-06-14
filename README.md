# Non-TAG → Pseudo-TAG Textualization Pipeline

A research pipeline that takes a graph with **no node text** (edges + labels only) and generates synthetic styled text per node, producing a pseudo-TAG graph compatible with LLM-based node classifiers (GraphGPT, LLaGA, GraphPrompter).

The goal is to study whether the **narrative style** of generated text (poetry / news / story) affects downstream classification accuracy, with class-to-topic mappings held fixed across conditions so that style is the only variable.

---

## How It Works

Each node's text is generated from the label distribution of its most informative neighbors, selected via Personalized PageRank. Class labels are remapped to literary themes (e.g. `Diabetes_Mellitus_Type_2 → the harvest`) at generation time, so the model learns an implicit association rather than matching class names literally.

**Leakage rule (non-negotiable):** a node's text may only draw on the labels of its train/val neighbors. Test node labels are never used during generation.

The pipeline runs in three phases:

```
Phase 1 — run_precompute.py    PPR selection + class proportions     (no API calls)
Phase 2 — run_textualize.py    topic mapping + LLM text generation   (uses API key)
Phase 3 — run_encode.py        encode raw_texts → node features x    (local model)
```

Phases are fully decoupled: changing narrative topics only requires rerunning Phase 2. Changing the encoder only requires rerunning Phase 3.

---

## Project Structure

```
nontag_pipeline/
  config.py           all tuneable constants (dataset, style, LLM settings, paths)
  data.py             load_dataset() — Planetoid download + 60/20/20 seeded split
  select.py           ppr_selection() + neighbor_label_proportions()
  narratives.py       TOPIC_MAPS (class → theme) + STYLE_TEMPLATES
  llm.py              LLM calls (OpenAI / Ollama) with SHA-256 disk cache and retry
  textualize.py       build_generation_prompt() + generate_node_text()
  io.py               save_pseudo_tag() — writes PyG .pt + human-readable .json
  encode/
    sbert.py          SBERT encoder — all-mpnet-base-v2 (768-dim)   → GraphPrompter
    graphgpt.py       Transformer mean-pool encoder (768-dim)        → GraphGPT

run_precompute.py     Phase 1 driver
run_textualize.py     Phase 2 driver
run_encode.py         Phase 3 driver

tests/                55 unit tests (pytest)
outputs/              generated files (gitignored)
data/                 Planetoid download cache (gitignored)
.cache/llm/           LLM response cache, keyed by SHA-256 (gitignored)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running

```bash
# Set your API key (OpenAI, or any compatible server)
export LLM_API_KEY="sk-..."

# Phase 1 — PPR + class proportions (no API calls, style-independent)
python run_precompute.py --n-train 1000    # 1000 train + proportional val/test
python run_precompute.py                   # all nodes (~19k for PubMed)

# Phase 2 — map proportions to topics, generate text, save pseudo-TAG
python run_textualize.py --n-train 1000
python run_textualize.py --style news      # change style without rerunning Phase 1

# Phase 3 — encode raw_texts into node feature vectors
python run_encode.py --encoder sbert       # GraphPrompter
python run_encode.py --encoder graphgpt    # GraphGPT

# Run tests
python -m pytest tests/ -v
```

**Changing narrative topics** (e.g. editing `TOPIC_MAPS` in `narratives.py`): only Phase 2 needs to rerun — Phase 1 precomputed proportions are topic-independent.

**Comparing styles**: run Phase 2 once per style with `--style poetry/news/story`. Topics stay fixed; only the rendering changes. Each run writes a separate output file.

---

## Output Files

| File | Description |
|---|---|
| `outputs/precomputed_<dataset>[_N].json` | PPR selections + class proportions (Phase 1) |
| `outputs/pseudo_tag_<dataset>_<style>[_N].pt` | PyG Data with `raw_texts` (Phase 2) |
| `outputs/pseudo_tag_<dataset>_<style>[_N].json` | Human-readable version — do not feed to predictor |
| `outputs/pseudo_tag_<dataset>_<style>[_N]_<encoder>.pt` | PyG Data with `raw_texts` + `x` (Phase 3) |

The `.pt` files with `x` are the direct input to GraphGPT / GraphPrompter GNN components:

```python
data = torch.load("outputs/pseudo_tag_pubmed_poetry_sbert.pt", weights_only=False)
data.x            # float tensor (N, 768) — node features from SBERT
data.raw_texts    # list[str | None] — generated text per node
data.y            # node labels
data.train_mask / data.val_mask / data.test_mask
data.edge_index   # standard PyG format
data.class_names
```

---

## Configuration

All settings live in `nontag_pipeline/config.py`. Nothing else needs to change.

| Variable | Default | Options |
|---|---|---|
| `DATASET` | `"pubmed"` | `"pubmed"`, `"cora"` |
| `STYLE` | `"poetry"` | `"poetry"`, `"news"`, `"story"` |
| `LLM_BACKEND` | `"openai"` | `"openai"`, `"ollama"` |
| `LLM_MODEL` | `"gpt-4o-mini"` | any model name |
| `LLM_BASE_URL` | OpenAI endpoint | any compatible API URL |
| `LLM_KEY_ENV` | `"LLM_API_KEY"` | name of the env var holding your key |
| `PPR_K` | `2` | k-hop limit for PPR |
| `PPR_M` | `10` | top-m neighbors to select |

**Switching providers:** set `LLM_KEY_ENV = "UNIVERSITY_API_KEY"` (or whatever) and `LLM_BASE_URL` to your server — no code changes.

---

## Topic Mappings

Class labels are remapped to themes at generation time (Phase 2, not Phase 1). Topics are chosen to evoke the real-world meaning of each class so the generated texts are semantically distinct — making the predictor's classification task meaningful.

**PubMed**

| Class | Theme | Rationale |
|---|---|---|
| Diabetes_Mellitus_Experimental | the laboratory | lab-induced models, controlled interventions |
| Diabetes_Mellitus_Type_1 | the siege | immune system besieging and destroying beta cells |
| Diabetes_Mellitus_Type_2 | the harvest | metabolic accumulation, gradual insulin resistance |

**Cora**

| Class | Theme | Rationale |
|---|---|---|
| Case_Based | the archive | reasoning from stored past examples |
| Genetic_Algorithms | evolution | natural selection, mutation, survival of fittest solutions |
| Neural_Networks | the mind | brain-inspired layered computation |
| Probabilistic_Methods | the fog | uncertainty, degrees of belief |
| Reinforcement_Learning | the arena | agents, reward signals, trial and error |
| Rule_Learning | the ledger | explicit symbolic rules, deterministic inference |
| Theory | the cosmos | abstract proofs, mathematical foundations |

---

## Roadmap

- [x] Data loading + 60/20/20 seeded split
- [x] PPR neighbor selection
- [x] Theme-based LLM text generation (poetry / news / story)
- [x] Pseudo-TAG save (PyG .pt + .json)
- [x] SBERT encoding → node features (GraphPrompter)
- [x] Transformer encoding → node features (GraphGPT)
- [ ] GraphGPT / LLaGA / GraphPrompter integration
- [ ] Accuracy comparison across narrative styles
