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
