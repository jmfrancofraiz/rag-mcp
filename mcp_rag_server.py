#!/usr/bin/env python3

import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Reuse logic from rag_cli
from rag_cli import (
    _detect_embeddings_provider,
    _detect_chat_provider,
    _ensure_keys_for_provider,
    _make_embeddings,
    build_rag_chain,
)
from langchain_community.vectorstores import Chroma

# FastMCP server
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("rag")


def _perform_query(
    question: str,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    embedding_model: Optional[str] = None,
) -> str:
    # Providers and keys
    emb_provider = _detect_embeddings_provider()
    chat_provider = _detect_chat_provider()
    _ensure_keys_for_provider(emb_provider)
    _ensure_keys_for_provider(chat_provider)

    # Defaults
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_persist = os.path.join(repo_root, ".chroma")

    persist_dir = persist_dir or os.environ.get("RAG_PERSIST_DIR", default_persist)
    collection_name = collection_name or os.environ.get("RAG_COLLECTION_NAME", "wiki-md")
    model = model or (
        os.environ.get("RAG_GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
        if os.environ.get("GOOGLE_API_KEY")
        else os.environ.get("RAG_CHAT_MODEL", "gpt-4o-mini")
    )
    try:
        temperature = float(temperature) if temperature is not None else float(os.environ.get("RAG_TEMPERATURE", 0.0))
    except ValueError:
        temperature = 0.0
    embedding_model = embedding_model or os.environ.get("RAG_EMBEDDING_MODEL", "text-embedding-3-small")

    if not question:
        return "ERROR: question is required"

    if not os.path.isdir(persist_dir):
        return f"ERROR: persist_dir does not exist or is not a directory: {persist_dir}"

    # Build vector store and chain
    embeddings = _make_embeddings(emb_provider, embedding_model)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name=collection_name,
    )

    chain, retriever = build_rag_chain(
        vector_store=vector_store, chat_provider=chat_provider, model=model, temperature=temperature
    )

    # Run retrieval for sources and answer
    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)

    # Collect sources
    sources = []
    seen = set()
    for d in retrieved_docs:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        if src not in seen:
            seen.add(src)
            sources.append(str(src))

    # Compose response text
    lines = ["=== Answer ===", answer]
    if sources:
        lines.append("\n=== Sources ===")
        lines.extend(f"- {s}" for s in sources)
    result_text = "\n".join(lines)

    return result_text


@mcp.tool
def ping() -> str:
    """Health check tool to verify server connectivity."""
    return "pong"


@mcp.tool
def query(
    question: str,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    embedding_model: Optional[str] = None,
) -> str:
    """Query the RAG index and return the answer plus sources."""
    return _perform_query(
        question=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model=model,
        temperature=temperature,
        embedding_model=embedding_model,
    )


if __name__ == "__main__":
    print("rag-mcp-server is up. Listening on stdio for MCP requests...", file=sys.stderr, flush=True)
    # FastMCP defaults to stdio transport
    mcp.run()
