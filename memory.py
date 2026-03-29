# ============================================================
# memory.py — Long-term vector memory using ChromaDB
# ============================================================
# WHY VECTOR MEMORY?
# A regular list just stores text. Vector memory converts text into
# embeddings (arrays of numbers that capture meaning), then finds
# the most SEMANTICALLY SIMILAR past memories when you query it.
#
# Example: If you stored "transformer models use attention mechanisms"
# and later query "how does self-attention work?", vector memory will
# retrieve that note even though the exact words don't match.
#
# HOW IT WORKS:
# 1. Text → embedding (via a small local model in ChromaDB)
# 2. Embedding stored in a vector database (ChromaDB)
# 3. At query time: your query → embedding → find nearest vectors
# 4. Return the original text of those nearest vectors
# ============================================================

import chromadb
from chromadb.utils import embedding_functions
from config import MEMORY_COLLECTION_NAME, MEMORY_TOP_K, CHROMA_PERSIST_DIR
import uuid
import json
from datetime import datetime


class AgentMemory:
    """
    Manages long-term semantic memory for the research agent.

    Memory persists to disk (CHROMA_PERSIST_DIR) so the agent can
    recall findings from previous research sessions too.
    """

    def __init__(self):
        # PersistentClient saves memory to disk between runs.
        # Use chromadb.Client() instead if you want in-memory only.
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # ChromaDB's built-in sentence transformer for embeddings.
        # Uses "all-MiniLM-L6-v2" — a small, fast, local model.
        # No API key needed, runs entirely on your machine.
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()

        # Get or create the collection (like a table in a database)
        self.collection = self.client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}  # cosine similarity for text
        )

        print(f"[Memory] Initialized. Collection has {self.collection.count()} existing memories.")

    def store(self, content: str, metadata: dict = None) -> str:
        """
        Store a piece of information in long-term memory.

        Args:
            content: The text to remember
            metadata: Optional dict with extra info (topic, session_id, etc.)

        Returns:
            The unique ID assigned to this memory
        """
        memory_id = str(uuid.uuid4())

        # Add timestamp and any custom metadata
        full_metadata = {
            "timestamp": datetime.now().isoformat(),
            "content_preview": content[:100]
        }
        if metadata:
            full_metadata.update(metadata)

        self.collection.add(
            documents=[content],
            metadatas=[full_metadata],
            ids=[memory_id]
        )

        return memory_id

    def retrieve(self, query: str, top_k: int = None) -> list[dict]:
        """
        Retrieve the most relevant memories for a given query.

        ChromaDB automatically converts both the stored texts and the
        query to embeddings and finds the closest matches.

        Args:
            query: What you're looking for (plain English)
            top_k: How many results to return (defaults to config value)

        Returns:
            List of dicts with 'content', 'similarity', and 'metadata'
        """
        if top_k is None:
            top_k = MEMORY_TOP_K

        # Can't query more than what's stored
        count = self.collection.count()
        if count == 0:
            return []

        top_k = min(top_k, count)

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            # Distance → similarity (cosine distance: 0 = identical, 2 = opposite)
            distance = results["distances"][0][i]
            similarity = 1 - (distance / 2)  # normalize to 0-1

            memories.append({
                "content": doc,
                "similarity": round(similarity, 3),
                "metadata": results["metadatas"][0][i]
            })

        # Sort by similarity descending
        memories.sort(key=lambda x: x["similarity"], reverse=True)
        return memories

    def retrieve_as_context(self, query: str) -> str:
        """
        Retrieve memories and format them as a readable context string
        that can be injected into the LLM prompt.

        Returns:
            Formatted string of relevant past memories, or empty string
        """
        memories = self.retrieve(query)
        if not memories:
            return ""

        lines = ["[Relevant past research from memory:]"]
        for i, m in enumerate(memories, 1):
            sim_pct = int(m["similarity"] * 100)
            lines.append(f"  {i}. (relevance: {sim_pct}%) {m['content']}")

        return "\n".join(lines)

    def store_session_notes(self, notes: list[dict]) -> None:
        """
        Bulk store all session notes into long-term memory at the end
        of a research session so they persist for future sessions.

        Args:
            notes: List of {content, tag} dicts from the session
        """
        for note in notes:
            self.store(
                content=note["content"],
                metadata={"tag": note.get("tag", "general"), "source": "session_note"}
            )
        print(f"[Memory] Stored {len(notes)} session notes to long-term memory.")

    def clear(self) -> None:
        """Clear all memories. Useful for testing."""
        self.client.delete_collection(MEMORY_COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=MEMORY_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )
        print("[Memory] All memories cleared.")

    def count(self) -> int:
        """Return total number of stored memories."""
        return self.collection.count()
