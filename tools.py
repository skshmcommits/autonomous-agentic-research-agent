# ============================================================
# tools.py — All tools the agent can call
# ============================================================
# A "tool" is just a Python function the agent is allowed to use.
# Each tool has:
#   - A name (string identifier)
#   - A description (the LLM reads this to decide when to use it)
#   - A function (the actual code that runs)
#   - A parameters schema (tells the LLM what inputs to provide)
#
# The LLM never directly calls these — it outputs a structured
# JSON saying "call tool X with these arguments", and our
# agent.py reads that and calls the actual function.
# ============================================================

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from config import TAVILY_API_KEY, MAX_SEARCH_RESULTS

# Initialize Tavily client once (reused across calls)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


# ── Tool 1: Web Search ─────────────────────────────────────────────────────────

def search_web(query: str) -> dict:
    """
    Search the web using Tavily and return structured results.

    Tavily is an API built specifically for AI agents — unlike Google,
    it returns clean, pre-processed text that's easy for LLMs to read.

    Args:
        query: The search query string

    Returns:
        A dict with 'results' (list of {title, url, content}) and 'query'
    """
    try:
        response = tavily_client.search(
            query=query,
            max_results=MAX_SEARCH_RESULTS,
            search_depth="advanced",    # deeper search, more content per result
            include_raw_content=False   # processed content is cleaner
        )

        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", "No title"),
                "url": r.get("url", ""),
                "content": r.get("content", "")[:1500]  # cap at 1500 chars per result
            })

        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }

    except Exception as e:
        return {"error": str(e), "query": query, "results": []}


# ── Tool 2: Read URL ───────────────────────────────────────────────────────────

def read_url(url: str) -> dict:
    """
    Fetch and extract clean text from a specific URL.

    When the agent wants to read a full article (not just the snippet),
    it calls this tool with the URL. We use BeautifulSoup to strip all
    HTML tags and return just the readable text.

    Args:
        url: Full URL including https://

    Returns:
        A dict with 'url', 'title', and 'content' (plain text)
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script and style tags — they contain no readable content
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get the page title
        title = soup.title.string.strip() if soup.title else "No title"

        # Extract all paragraph text and join it
        paragraphs = soup.find_all("p")
        content = " ".join(p.get_text(strip=True) for p in paragraphs)

        # Cap at 3000 characters to avoid overwhelming the LLM context
        content = content[:3000] + ("..." if len(content) > 3000 else "")

        return {
            "url": url,
            "title": title,
            "content": content
        }

    except requests.exceptions.Timeout:
        return {"error": "Request timed out", "url": url, "content": ""}
    except requests.exceptions.RequestException as e:
        return {"error": str(e), "url": url, "content": ""}
    except Exception as e:
        return {"error": str(e), "url": url, "content": ""}


# ── Tool 3: Save Note ──────────────────────────────────────────────────────────

def save_note(content: str, tag: str = "general") -> dict:
    """
    Save an important finding to short-term session memory.

    This is separate from ChromaDB long-term memory. Think of this as
    the agent writing a sticky note to itself during a research session.
    These notes are included in the agent's context at each step so it
    doesn't forget key findings from earlier in the session.

    Args:
        content: The finding or insight to save
        tag: A label like "key_finding", "statistic", "definition", "gap"

    Returns:
        Confirmation dict
    """
    # The actual storage happens in agent.py via session_notes list.
    # This function just returns the data in a consistent format.
    return {
        "saved": True,
        "tag": tag,
        "content": content,
        "message": f"Note saved with tag '{tag}'. You can reference this later."
    }


# ── Tool 4: Finish Research ────────────────────────────────────────────────────

def finish_research(summary: str) -> dict:
    """
    Signal that research is complete and provide the final summary.

    The agent calls this when it believes it has gathered enough
    information to write a complete report. This terminates the
    ReAct loop and triggers report generation.

    Args:
        summary: A brief summary of what was found (1-3 sentences)

    Returns:
        Dict with 'done': True — the agent loop checks for this
    """
    return {
        "done": True,
        "summary": summary,
        "message": "Research phase complete. Generating final report now."
    }


# ── Tool Registry ──────────────────────────────────────────────────────────────
# This is what gets passed to the LLM. Each entry tells the model:
# what the tool is called, what it does, and what parameters to pass.
# The LLM reads these descriptions to decide which tool to use.

TOOL_REGISTRY = {
    "search_web": {
        "function": search_web,
        "description": "Search the web for information on a topic. Use this to find sources, facts, recent developments, and data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific for better results. E.g., 'graph neural networks survey 2024' not just 'neural networks'"
                }
            },
            "required": ["query"]
        }
    },
    "read_url": {
        "function": read_url,
        "description": "Read the full text content of a specific URL/webpage. Use this when you want more detail from a source than what the search snippet provided.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to read, e.g., 'https://example.com/article'"
                }
            },
            "required": ["url"]
        }
    },
    "save_note": {
        "function": save_note,
        "description": "Save an important finding, statistic, definition, or insight to memory so you can reference it later in your report.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The finding or insight to remember"
                },
                "tag": {
                    "type": "string",
                    "description": "Category label: 'key_finding', 'statistic', 'definition', 'gap', 'source', or 'general'",
                    "enum": ["key_finding", "statistic", "definition", "gap", "source", "general"]
                }
            },
            "required": ["content", "tag"]
        }
    },
    "finish_research": {
        "function": finish_research,
        "description": "Call this when you have gathered sufficient information and are ready to write the final report. Provide a brief summary of your findings.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A 2-3 sentence summary of the key findings from your research"
                }
            },
            "required": ["summary"]
        }
    }
}


# ── Schema Builder ─────────────────────────────────────────────────────────────
# Converts our TOOL_REGISTRY into the exact JSON format the Groq/OpenAI
# API expects for its "tools" parameter. This is boilerplate that the
# LLM provider requires to understand what tools are available.

def get_tools_schema() -> list:
    """Build the tools schema list for the LLM API call."""
    schema = []
    for tool_name, tool_info in TOOL_REGISTRY.items():
        schema.append({
            "type": "function",
            "function": {
                "name": tool_name,
                "description": tool_info["description"],
                "parameters": tool_info["parameters"]
            }
        })
    return schema
