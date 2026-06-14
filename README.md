# Non-TAG → Pseudo-TAG Textualization Pipeline

This project studies whether the **narrative style** of synthetically generated text affects the accuracy of LLM-based graph node classifiers (GraphPrompter, GraphGPT, LLaGA).

We start from a graph that has **no node text** — only edges and labels — and generate a short text for each node. The result is a *pseudo-TAG* graph that can be fed directly into text-attributed graph (TAG) classifiers. By generating the same graph three times with different styles (poetry / news / story) and comparing downstream accuracy, we isolate the effect of narrative style on classification.

---

## Method

For each node, we:
1. Select its most informative neighbors using Personalized PageRank (PPR)
2. Compute the proportion of each class among those neighbors
3. Map class labels to abstract literary themes (e.g. `Diabetes_Mellitus_Type_1 → war`)
4. Prompt an LLM to write a short text that evokes those themes in the chosen style

The generated text reflects the node's local graph context — not its own label. Test node labels are never used during generation.

**Datasets:** PubMed (3 classes), Cora (7 classes)
**Styles:** `poetry`, `news`, `story`

---

## Pipeline

The pipeline runs in three sequential steps:

```
Step 1 — precompute    PPR selection + class proportions  (no API calls)
Step 2 — textualize    LLM text generation                (requires API key)
Step 3 — embed         encode text → node feature vectors (local model)
```

**Install dependencies** (Python 3.10+):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Step 1 — Precompute** (run once per dataset):
```bash
python pipeline/precompute.py                  # all nodes
python pipeline/precompute.py --n-train 1000   # subset: 1000 train + proportional val/test
```

**Step 2 — Textualize** (run once per style):
```bash
export LLM_API_KEY="sk-..."
python pipeline/textualize.py --style story
python pipeline/textualize.py --style news
python pipeline/textualize.py --style poetry
```
Step 1 does not need to rerun between styles. Each style produces a separate output file.

**Step 3 — Embed**:
```bash
python pipeline/embed.py --embedder sbert     # for GraphPrompter
python pipeline/embed.py --embedder graphgpt  # for GraphGPT
```

---

## Outputs

All files are written to `outputs/`:

| File | Content |
|---|---|
| `precomputed_<dataset>.json` | PPR selections + class proportions (Step 1) |
| `pseudo_tag_<dataset>_<style>.pt` | PyG Data with `raw_texts` (Step 2) |
| `pseudo_tag_<dataset>_<style>_<model>.pt` | PyG Data with `raw_texts` + `x` (Step 3) |

The `.pt` file after Step 3 contains everything a TAG classifier needs:
```python
data = torch.load("outputs/pseudo_tag_pubmed_story_all-mpnet-base-v2.pt", weights_only=False)
data.x           # float32 (N, 768) — node feature vectors
data.raw_texts   # list[str | None]  — generated text per node
data.y           # node labels
data.edge_index  # graph structure (PyG format)
data.train_mask / data.val_mask / data.test_mask
data.class_names
```

---

## Using with GraphPrompter / GraphGPT

After Step 3, run the export script to produce the exact directory layout each model expects:

**GraphPrompter:**
```bash
python export/graphprompter.py --style story
```
This writes to `outputs/graphprompter/tape_<dataset>_<style>_<model>/`. Copy that folder into the cloned GraphPrompter repo under `dataset/`.

**GraphGPT / LLaGA:** *(integration not yet implemented)*

---

## Configuration

All settings are in `nontag_pipeline/config.py`. Key options:

| Variable | Default | What it controls |
|---|---|---|
| `DATASET` | `"pubmed"` | `"pubmed"` or `"cora"` |
| `STYLE` | `"story"` | `"poetry"`, `"news"`, `"story"` |
| `LLM_MODEL` | `"gpt-4o-mini"` | any OpenAI-compatible model |
| `LLM_BASE_URL` | OpenAI endpoint | swap to university server or Ollama |
| `LLM_KEY_ENV` | `"LLM_API_KEY"` | name of the env var holding the API key |

To use a university server or Ollama, change `LLM_BASE_URL` (and `LLM_BACKEND = "ollama"` for Ollama) — no other code changes needed.

---

## Tests

```bash
python -m pytest tests/ -v
```
