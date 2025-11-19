#!/usr/bin/env python3

import os
import sys
from typing import Optional

from dotenv import load_dotenv

# Reuse the CLI's query logic directly
from rag_cli import build_parser, query_command

# FastMCP server
from fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("TAB RAG")


def _perform_query(
    question: str,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    embedding_model: Optional[str] = None,
) -> str:
    # Delegate to the CLI's query_command, reusing its defaults via build_parser
    if not question:
        return "ERROR: --question is required for query command"

    import io
    import contextlib

    parser = build_parser()

    argv = [
        "query",
        "--question",
        str(question),
    ]
    if persist_dir:
        argv += ["--persist-dir", str(persist_dir)]
    if collection_name:
        argv += ["--collection-name", str(collection_name)]
    if model:
        argv += ["--model", str(model)]
    if temperature is not None:
        argv += ["--temperature", str(temperature)]
    if embedding_model:
        argv += ["--embedding-model", str(embedding_model)]

    # Capture stdout/stderr so we can return a string result
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            args = parser.parse_args(argv)
            # Call the same function used by the CLI subcommand
            query_command(args)
    except SystemExit as e:
        # Convert CLI-style exits into string errors for MCP clients
        std_err = err_buf.getvalue().strip()
        if std_err:
            return std_err
        std_out = out_buf.getvalue().strip()
        if std_out:
            return std_out
        return f"ERROR: query failed (exit {getattr(e, 'code', 1)})"

    std_out = out_buf.getvalue().strip()
    std_err = err_buf.getvalue().strip()
    if std_out:
        return std_out
    if std_err:
        return std_err
    return ""


@mcp.tool
def query(
    question: str,
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    embedding_model: Optional[str] = None,
) -> str:
    """Query TAB's RAG index and return the answer plus sources."""
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
