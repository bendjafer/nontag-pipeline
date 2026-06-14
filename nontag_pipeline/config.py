SEED: int = 42
DATASET: str = "pubmed"        # "cora" | "pubmed"
STYLE: str = "poetry"          # rendering only: "poetry" | "news" | "story" (topics are fixed)

TARGET_LEN: int = 80           # target word count for generated text

PPR_K: int = 2                 # k-hop neighbourhood limit for PPR
PPR_M: int = 10                # top-m neighbours to select
PPR_ALPHA: float = 0.85

LLM_BACKEND: str = "openai"    # "openai" | "ollama"
LLM_MODEL: str = "gpt-4o-mini"
LLM_BASE_URL: str = "https://api.openai.com/v1"
LLM_TEMPERATURE: float = 0.0   # deterministic generation (with SEED as API seed)

# Name of the environment variable that holds the API key.
# Export both keys and flip this line to switch — no code changes needed.
#   OpenAI:     LLM_KEY_ENV = "OPENAI_API_KEY"
#   University: LLM_KEY_ENV = "UNIVERSITY_API_KEY"
LLM_KEY_ENV: str = "LLM_API_KEY"


CACHE_DIR: str = ".cache/llm"
OUTPUT_DIR: str = "outputs"
DATA_ROOT: str = "data/planetoid"   # in-repo (gitignored) so it survives reboots
