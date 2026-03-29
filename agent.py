# ============================================================
# agent.py — The core Research Agent with the ReAct loop
# ============================================================
# THE REACT LOOP (Reason → Act → Observe):
#
#   REASON:  The LLM looks at the task + history and decides
#            what to do next (which tool to call and why)
#
#   ACT:     Our code executes the chosen tool and gets a result
#
#   OBSERVE: The result is fed back to the LLM as an observation
#            so it can reason about what to do next
#
# This repeats until the agent calls finish_research() or hits
# MAX_ITERATIONS (a safety limit to prevent infinite loops).
#
# VISUAL:
#   Task → [REASON → ACT → OBSERVE] → [REASON → ACT → OBSERVE] → ... → Report
# ============================================================

import json
import os
from groq import Groq # pyright: ignore[reportMissingImports]
from config import (
    GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    MAX_ITERATIONS, REPORTS_DIR
)
from tools import TOOL_REGISTRY, get_tools_schema
from memory import AgentMemory
from datetime import datetime


class ResearchAgent:
    """
    An autonomous research agent that uses the ReAct pattern to
    iteratively search, read, and synthesize information into reports.
    """

    def __init__(self):
        # Initialize the Groq LLM client
        self.llm = Groq(api_key=GROQ_API_KEY)

        # Initialize long-term vector memory
        self.memory = AgentMemory()

        # Short-term session notes (cleared each new research task)
        # These are notes the agent saves to itself mid-session
        self.session_notes: list[dict] = []

        # Full conversation history sent to the LLM each iteration
        # (This is how the LLM "remembers" what happened earlier in the loop)
        self.messages: list[dict] = []

        # Tools schema — tells the LLM what tools it can call
        self.tools_schema = get_tools_schema()

        # Ensure reports directory exists
        os.makedirs(REPORTS_DIR, exist_ok=True)

        print("[Agent] Research Agent initialized and ready.")

    # ── System Prompt ──────────────────────────────────────────────────────────

    def _build_system_prompt(self, topic: str) -> str:
        """
        The system prompt defines the agent's role, personality,
        and instructions. This is the most important prompt in the system.
        """
        return f"""You are an autonomous research agent. IMPORTANT: Always call a tool immediately. Never write explanatory text before calling a tool — just call the tool directly.

TOPIC: {topic}

YOUR RESEARCH PROCESS:
1. Start by breaking the topic into key sub-questions you need to answer
2. Use search_web to find relevant sources and information
3. Use read_url on the most promising URLs to get full article content
4. Use save_note to preserve key findings, statistics, and insights
5. Keep searching until you have covered: background, current state, key findings, debates/gaps, and future directions
6. Once you have sufficient depth (at least 5-7 solid sources), call finish_research

RULES:
- Always be specific in search queries (add year, domain, context)
- Don't repeat the same search query twice
- If a URL fails, move on to the next source
- Save important statistics and facts as notes so you don't forget them
- Track what you've already covered to avoid repetition
- Aim for depth over breadth — fully understand a few key aspects rather than superficially touching many

QUALITY BAR:
- Minimum 5 different search queries covering different aspects
- Minimum 3 full articles read via read_url
- Minimum 5 notes saved
- Cover: definition/background, current research, applications, limitations, open questions
"""

    # ── Tool Execution ─────────────────────────────────────────────────────────

    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict:
        """
        Execute a tool and return its result.
        Also handles the special 'save_note' tool to store in session memory.
        """
        if tool_name not in TOOL_REGISTRY:
            return {"error": f"Unknown tool: {tool_name}"}

        tool_fn = TOOL_REGISTRY[tool_name]["function"]
        result = tool_fn(**tool_args)

        # If the agent saved a note, store it in our session list
        if tool_name == "save_note" and result.get("saved"):
            self.session_notes.append({
                "content": tool_args["content"],
                "tag": tool_args.get("tag", "general")
            })

        return result

    # ── Session Notes Formatter ────────────────────────────────────────────────

    def _format_session_notes(self) -> str:
        """Format current session notes for injection into the LLM prompt."""
        if not self.session_notes:
            return ""

        lines = ["\n[YOUR SESSION NOTES SO FAR:]"]
        for i, note in enumerate(self.session_notes, 1):
            lines.append(f"  [{note['tag'].upper()}] {note['content']}")
        return "\n".join(lines)

    # ── Single ReAct Step ──────────────────────────────────────────────────────

    def _step(self, iteration: int) -> tuple[bool, dict | None]:
            print(f"\n[Agent] --- Iteration {iteration} ---")

            # Build action prompt — model outputs JSON instead of using tool API
            action_prompt = """You must respond with ONLY a JSON object in exactly this format, nothing else:

        {"tool": "search_web", "args": {"query": "your query here"}}

        Available tools:
        - search_web: {"tool": "search_web", "args": {"query": "string"}}
        - read_url: {"tool": "read_url", "args": {"url": "string"}}
        - save_note: {"tool": "save_note", "args": {"content": "string", "tag": "key_finding|statistic|definition|gap|source|general"}}
        - finish_research: {"tool": "finish_research", "args": {"summary": "string"}}

        Respond with ONE JSON object only. No explanation, no markdown, no code blocks. Just the raw JSON."""

            augmented_messages = list(self.messages)
            if self.session_notes:
                notes = "\n".join(f"[{n['tag'].upper()}] {n['content']}" for n in self.session_notes)
                augmented_messages.append({
                    "role": "user",
                    "content": f"Your notes so far:\n{notes}\n\n{action_prompt}"
                })
            else:
                augmented_messages.append({
                    "role": "user", 
                    "content": action_prompt
                })

            # Call LLM without tools API — just plain text response
            response = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=augmented_messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=500
            )

            raw = response.choices[0].message.content.strip()
            print(f"[Agent Raw Output] {raw[:200]}")

            # Parse the JSON action
            try:
                # Strip markdown code blocks if model added them anyway
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                action = json.loads(raw.strip())
            except json.JSONDecodeError:
                print("[Agent] Could not parse JSON, retrying next iteration...")
                self.messages.append({"role": "assistant", "content": raw})
                return False, None

            tool_name = action.get("tool")
            tool_args = action.get("args", {})

            print(f"[Tool Call] {tool_name}({json.dumps(tool_args)[:100]})")

            # Store in history as plain text
            self.messages.append({"role": "assistant", "content": raw})

            # Execute the tool
            result = self._execute_tool(tool_name, tool_args)

            if tool_name == "finish_research" and result.get("done"):
                print("[Agent] Research phase complete!")
                return True, result

            result_preview = str(result)[:300]
            print(f"[Tool Result] {result_preview}")

            # Feed result back as user message
            self.messages.append({
                "role": "user",
                "content": f"Tool result for {tool_name}:\n{json.dumps(result)[:1000]}\n\nContinue researching."
            })

            return False, None

            # ── Report Generator ───────────────────────────────────────────────────────

    def _generate_report(self, topic: str, research_summary: str) -> str:
                """
                After the ReAct loop finishes, call the LLM one more time
                with all the gathered notes and context to write a polished report.
                """
                print("\n[Agent] Generating final report...")

                notes_text = "\n".join(
                    f"- [{n['tag'].upper()}] {n['content']}"
                    for n in self.session_notes
                )

                # Build a fresh prompt for report generation
                report_prompt = f"""You are a research report writer. Based on the research notes below, write a comprehensive, well-structured research report on: "{topic}"

        RESEARCH NOTES COLLECTED:
        {notes_text if notes_text else "See conversation history for gathered information."}

        RESEARCH SUMMARY:
        {research_summary}

        REPORT FORMAT:
        # {topic} — Research Report

        ## Executive Summary
        (2-3 paragraph overview of key findings)

        ## Background & Context
        (History, definitions, foundational concepts)

        ## Current State of the Field
        (Latest developments, major players, recent work)

        ## Key Findings
        (Most important insights from the research)

        ## Applications & Use Cases
        (Real-world applications and examples)

        ## Limitations & Open Questions
        (What's unknown, debated, or challenging)

        ## Future Directions
        (Where the field is heading)

        ## Sources & References
        (List all sources found during research)

        ---
        Write in a clear, academic but accessible tone. Be specific — include numbers, dates, and names where available. Target length: 800-1200 words.
        """

                response = self.llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
        {
            "role": "system",
            "content": "You are a research report writer. Write clear, structured, detailed reports."
        },
        {
            "role": "user",
            "content": report_prompt
        }
    ],
                    
                    temperature=0.4,
                    max_tokens=3000
                )

                return response.choices[0].message.content

            # ── Save Report to File ────────────────────────────────────────────────────

    def _save_report(self, topic: str, report: str) -> str:
        """Save the report to a markdown file and return the file path."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
        safe_topic = safe_topic.replace(" ", "_")[:50]
        filename = f"{safe_topic}_{timestamp}.md"
        filepath = os.path.join(REPORTS_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# Research Report: {topic}\n")
            f.write(f"*Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}*\n\n")
            f.write(report)

        return filepath

    # ── Main Research Method ───────────────────────────────────────────────────

    def research(self, topic: str) -> dict:
        """
        Main entry point. Run the full research pipeline on a topic.

        Args:
            topic: The research topic (e.g., "graph neural networks in drug discovery")

        Returns:
            Dict with 'report', 'filepath', 'iterations', 'notes_count'
        """
        print(f"\n{'='*60}")
        print(f"[Agent] Starting research on: {topic}")
        print(f"{'='*60}")

        # Reset state for new research session
        self.session_notes = []
        self.messages = []

        # Initialize the conversation with the system prompt + first user message
        self.messages.append({
            "role": "system",
            "content": self._build_system_prompt(topic)
        })
        self.messages.append({
            "role": "user",
            "content": f"Research this topic: {topic}. Call search_web now to begin."
        })

        # ── RUN THE REACT LOOP ─────────────────────────────────────────────────
        finish_result = None
        for iteration in range(1, MAX_ITERATIONS + 1):
            is_done, finish_result = self._step(iteration)

            if is_done:
                break

            if iteration == MAX_ITERATIONS:
                print(f"[Agent] Reached max iterations ({MAX_ITERATIONS}). Generating report with gathered data.")
                finish_result = {"summary": "Research completed via max iterations."}

        # ── GENERATE REPORT ────────────────────────────────────────────────────
        summary = finish_result.get("summary", "") if finish_result else ""
        report = self._generate_report(topic, summary)

        # ── SAVE TO DISK ───────────────────────────────────────────────────────
        filepath = self._save_report(topic, report)
        print(f"\n[Agent] Report saved to: {filepath}")

        # ── STORE SESSION TO LONG-TERM MEMORY ─────────────────────────────────
        if self.session_notes:
            self.memory.store_session_notes(self.session_notes)

        print(f"[Agent] Research complete! {len(self.session_notes)} notes collected.")

        return {
            "report": report,
            "filepath": filepath,
            "iterations_used": iteration,
            "notes_count": len(self.session_notes),
            "topic": topic
        }
