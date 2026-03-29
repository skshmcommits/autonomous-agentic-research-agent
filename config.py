# ============================================================
# config.py — Central configuration for the Research Agent
# ============================================================
# HOW TO USE:
# 1. Get a free Groq API key at: https://console.groq.com
# 2. Get a free Tavily API key at: https://app.tavily.com
# 3. Replace the placeholder strings below with your actual keys
# ============================================================

# --- API Keys ---
GROQ_API_KEY = "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"
TAVILY_API_KEY = "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"

# --- LLM Settings ---
# We use Groq because it's free and extremely fast.
# Model options: "llama-3.3-70b-versatile", "mixtral-8x7b-32768"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3      # Lower = more focused/deterministic
LLM_MAX_TOKENS = 2048

# --- Agent Settings ---
MAX_ITERATIONS = 15        # Max steps before the agent stops (prevents infinite loops)
MAX_SEARCH_RESULTS = 5     # How many search results Tavily returns per query

# --- Memory Settings ---
MEMORY_COLLECTION_NAME = "research_memory"   # ChromaDB collection name
MEMORY_TOP_K = 3                             # How many past memories to retrieve per step
CHROMA_PERSIST_DIR = "./chroma_db"           # Where ChromaDB saves data on disk

# --- Output Settings ---
REPORTS_DIR = "./reports"   # Folder where final reports are saved
