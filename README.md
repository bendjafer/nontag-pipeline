# Non-TAG → Pseudo-TAG Textualization Pipeline

A research pipeline that takes a graph with **no node text** (edges + labels only) and generates synthetic styled text per node, producing a pseudo-TAG graph compatible with LLM-based node classifiers (GraphGPT, LLaGA, GraphPrompter).

The goal is to study whether the **narrative style** of generated text (poetry / news / story) affects downstream classification accuracy, with class-to-topic mappings held fixed across conditions so that style is the only variable.

---

## How It Works

Each node's text is generated from the label distribution of its most informative neighbors, selected via Personalized PageRank. The class labels are remapped to literary themes (e.g. `Diabetes_Mellitus_Type_2 → nature`) before generation, so the model learns an implicit association rather than matching class names literally.

**Leakage rule (non-negotiable):** a node's text may only draw on the labels of its train/val neighbors. Test node labels are never used during generation.

The pipeline runs in two phases:

```
Phase 1 — run_precompute.py    PPR selection + theme proportions  (no API calls)
Phase 2 — run_textualize.py    LLM generation + save pseudo-TAG   (uses API key)
```

Separating the phases lets you inspect themes before spending API budget, and re-run generation with a different style without recomputing PPR.

---

## Project Structure

```
nontag_pipeline/
  config.py        all tuneable constants (dataset, style, LLM settings, paths)
  data.py          load_dataset() — Planetoid download + 60/20/20 seeded split
  select.py        ppr_selection() + neighbor_label_proportions()
  narratives.py    TOPIC_MAPS (class → theme) + STYLE_TEMPLATES + map_proportions_to_themes()
  llm.py           LLM calls (OpenAI / Ollama) with SHA-256 disk cache and retry
  textualize.py    build_generation_prompt() + generate_node_text()
  io.py            save_pseudo_tag() — writes PyG .pt + human-readable .json

run_precompute.py  Phase 1 driver
run_textualize.py  Phase 2 driver

tests/             45 unit tests (pytest)
outputs/           generated files (gitignored)
data/              Planetoid download cache (gitignored)
.cache/llm/        LLM response cache, keyed by SHA-256 (gitignored)
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

# Phase 1 — compute PPR + themes (no API calls)
python run_precompute.py --n-train 1000    # 1000 train + proportional val/test
python run_precompute.py                   # all nodes (~19k for PubMed)

# Phase 2 — generate text and save pseudo-TAG
python run_textualize.py --n-train 1000
python run_textualize.py

# Run tests
python -m pytest tests/ -v
```

Output files land in `outputs/`:
- `pseudo_tag_<dataset>_<style>[_<N>].pt` — PyG `Data` object with `raw_texts`, ready for any TAG predictor
- `pseudo_tag_<dataset>_<style>[_<N>].json` — human-readable version for inspection (contains all labels including test — do not feed to a predictor)

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

**Comparing styles:** run Phase 2 once per style. Topics stay fixed; only the rendering changes. Each run writes a separate output file.

---

## Output Format (.pt)

The `.pt` file is a PyG `Data` object:

```python
data = torch.load("outputs/pseudo_tag_pubmed_poetry.pt", weights_only=False)
data.raw_texts    # list[str | None], one per node; None = no neighbor signal
data.y            # node labels
data.train_mask   # bool tensor
data.val_mask
data.test_mask
data.class_names  # list of class name strings
data.edge_index   # standard PyG format
```

`raw_texts` is the attribute expected by GraphGPT, LLaGA, and GraphPrompter.

---

## Topic Mappings

Class labels are mapped to themes before generation and never named in the output, so downstream classification requires genuine semantic understanding rather than keyword matching.

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
- [x] Theme-based LLM text generation
- [x] Pseudo-TAG save (PyG .pt + .json)
- [ ] GraphGPT / LLaGA / GraphPrompter integration
- [ ] SBERT encoding of generated texts → node features
- [ ] Accuracy comparison across narrative styles
