# ============================================================
# main.py — Command-line interface for the Research Agent
# ============================================================
# HOW TO RUN:
#   python main.py
#   python main.py --topic "graph neural networks in drug discovery"
#   python main.py --topic "diffusion transformers" --clear-memory
# ============================================================

import argparse
import sys
import os
from agent import ResearchAgent


def print_banner():
    print("""
╔══════════════════════════════════════════════════════╗
║           🔍  Autonomous Research Agent              ║
║      ReAct Loop + Vector Memory + Web Search         ║
╚══════════════════════════════════════════════════════╝
""")


def print_result(result: dict):
    """Pretty print the research result."""
    print("\n" + "="*60)
    print("✅ RESEARCH COMPLETE")
    print("="*60)
    print(f"Topic:         {result['topic']}")
    print(f"Iterations:    {result['iterations_used']}")
    print(f"Notes saved:   {result['notes_count']}")
    print(f"Report saved:  {result['filepath']}")
    print("="*60)
    print("\n📄 REPORT PREVIEW (first 500 chars):")
    print("-"*60)
    print(result['report'][:500] + "...\n")
    print(f"Full report: {result['filepath']}")


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent — researches any topic and generates a structured report"
    )
    parser.add_argument(
        "--topic",
        type=str,
        help="Research topic (e.g., 'attention mechanisms in transformers')"
    )
    parser.add_argument(
        "--clear-memory",
        action="store_true",
        help="Clear all stored long-term memory before starting"
    )
    parser.add_argument(
        "--memory-stats",
        action="store_true",
        help="Show how many memories are stored and exit"
    )

    args = parser.parse_args()

    print_banner()

    # Initialize agent
    agent = ResearchAgent()

    # Handle --memory-stats flag
    if args.memory_stats:
        count = agent.memory.count()
        print(f"[Memory] Total memories stored: {count}")
        sys.exit(0)

    # Handle --clear-memory flag
    if args.clear_memory:
        agent.memory.clear()
        print("[Memory] Cleared all long-term memories.")

    # Get topic
    if args.topic:
        topic = args.topic
    else:
        print("Enter the topic you want to research.")
        print("Examples:")
        print("  - Diffusion Transformers for graph-structured data")
        print("  - Agentic AI systems and LLM orchestration")
        print("  - Graph neural networks in drug discovery")
        print()
        topic = input("Research topic: ").strip()

        if not topic:
            print("No topic provided. Exiting.")
            sys.exit(1)

    # Confirm before starting (research uses API calls)
    print(f"\nTopic: {topic}")
    confirm = input("Start research? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        sys.exit(0)

    # Run research
    try:
        result = agent.research(topic)
        print_result(result)

    except KeyboardInterrupt:
        print("\n\n[Agent] Research interrupted by user.")
        sys.exit(0)

    except Exception as e:
        print(f"\n[Error] Research failed: {e}")
        print("Common fixes:")
        print("  - Check your API keys in config.py")
        print("  - Ensure you have internet connection")
        print("  - Run: pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
