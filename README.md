# 🔍 Autonomous Research Agent

An autonomous AI research agent built from scratch using the **ReAct (Reason → Act → Observe)** pattern. Give it any topic and it will autonomously search the web, read sources, synthesize findings, and produce a structured research report.

---

## Architecture Overview

```
main.py          ← CLI entry point
agent.py         ← Core ReAct loop + report generation
tools.py         ← Web search, URL reader, note saver
memory.py        ← ChromaDB vector memory (long-term)
config.py        ← API keys and settings
reports/         ← Generated reports saved here
chroma_db/       ← ChromaDB persistent storage
```

---

## How It Works

### The ReAct Loop

```
TASK
 │
 ▼
[REASON] ─── LLM looks at history + memory, decides what to do next
    │
    ▼
[ACT] ────── Agent executes a tool (search_web / read_url / save_note)
    │
    ▼
[OBSERVE] ── Tool result fed back to LLM as context
    │
    ▼
 Loop back to REASON... until agent calls finish_research()
    │
    ▼
[GENERATE REPORT] ── Final LLM call synthesizes all notes into a report
```

### Tools Available to the Agent

| Tool | What it does |
|------|-------------|
| `search_web(query)` | Searches the web via Tavily API, returns top results |
| `read_url(url)` | Fetches full article text from a URL |
| `save_note(content, tag)` | Saves a finding to session memory |
| `finish_research(summary)` | Signals research is done, triggers report |

### Memory System

- **Short-term (session notes):** A running list of findings saved via `save_note`. Injected into the LLM context at each step so it doesn't forget earlier findings.
- **Long-term (ChromaDB):** All session notes are stored as vector embeddings at the end of a session. Future research sessions retrieve semantically similar past knowledge to bootstrap research.

---

## Setup

### Step 1: Get API Keys (both free)

1. **Groq** (LLM): https://console.groq.com → Create account → API Keys → Create new key
2. **Tavily** (Web search): https://app.tavily.com → Sign up → Get API Key

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Add your API keys

Open `config.py` and replace the placeholders:

```python
GROQ_API_KEY = "gsk_your_actual_key_here"
TAVILY_API_KEY = "tvly_your_actual_key_here"
```

### Step 4: Run

```bash
# Interactive mode (will prompt for topic)
python main.py

# Specify topic directly
python main.py --topic "diffusion transformers for graph data"

# Clear memory before starting
python main.py --topic "agentic AI" --clear-memory

# Check how many memories are stored
python main.py --memory-stats
```

---

## Example Output

```
[Agent] --- Iteration 1 ---
[Agent Reasoning] I'll start by searching for foundational concepts...
[Tool Call] search_web({"query": "diffusion transformers graph neural networks 2024"})
[Tool Result] {"results": [{"title": "DiT for graphs...", ...}]}

[Agent] --- Iteration 2 ---
[Tool Call] read_url({"url": "https://arxiv.org/abs/..."})
[Tool Result] {"title": "Graph Diffusion Transformer", "content": "..."}

[Agent] --- Iteration 3 ---
[Tool Call] save_note({"content": "DiT uses noise prediction on node features", "tag": "key_finding"})

... (continues for 10-15 iterations) ...

[Agent] Research phase complete!
[Agent] Generating final report...
[Agent] Report saved to: ./reports/diffusion_transformers_20250115_143022.md
```

---

## Project Structure Details

### `config.py`
Central configuration. Change `LLM_MODEL`, `MAX_ITERATIONS`, `MEMORY_TOP_K` here.

### `tools.py`
All tools the agent can call. To add a new tool:
1. Write the function
2. Add it to `TOOL_REGISTRY` with name, description, and parameters schema
3. That's it — the agent will automatically be able to use it

### `memory.py`
Wraps ChromaDB. Key methods:
- `store(content, metadata)` — Save text to vector DB
- `retrieve(query)` — Find semantically similar memories
- `retrieve_as_context(query)` — Formatted string for LLM injection

### `agent.py`
The brain. Key methods:
- `_step(iteration)` — One iteration of the ReAct loop
- `_execute_tool(name, args)` — Runs the tool and handles side effects
- `_generate_report(topic, summary)` — Final report synthesis
- `research(topic)` — Main entry point, runs everything

---

## Possible Extensions

- **Add a web UI** using Streamlit (`pip install streamlit`) with `streamlit run app.py`
- **Add more tools:** calculator, Python code executor, Wikipedia reader, arXiv paper fetcher
- **Multi-agent:** spawn sub-agents for parallel research on different subtopics
- **Export formats:** add PDF or DOCX report export
- **Evaluation:** score report quality automatically using an LLM judge

---

## Technologies Used

| Technology | Purpose | Why |
|-----------|---------|-----|
| Groq + Llama 3.3 70B | LLM backbone | Free, extremely fast inference |
| Tavily | Web search | Built specifically for AI agents, clean results |
| ChromaDB | Vector database | Local, no cloud needed, easy setup |
| BeautifulSoup | HTML parsing | Lightweight, reliable text extraction |
| Python | Language | |
