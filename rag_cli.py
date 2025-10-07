#!/usr/bin/env python3
"""
RAG CLI: Index markdown files into Chroma and query via OpenAI chat.

Subcommands:
  - index:  Load .md files, split, embed, persist to Chroma
  - query:  Query persisted Chroma and generate answers with context

Environment:
  - OPENAI_API_KEY must be set

Example usage:
  python rag_cli.py index --source-dir "/Users/jmffraiz/BAT-Repos/RAG/Consumer Platforms.wiki" \
                         --persist-dir "/Users/jmffraiz/BAT-Repos/RAG/.chroma"

  python rag_cli.py query --question "How do we deploy X?" \
                         --persist-dir "/Users/jmffraiz/BAT-Repos/RAG/.chroma"
"""

import argparse
import os
import sys
from typing import Iterable, List, Set

from dotenv import load_dotenv
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
)
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma


# Load environment variables from a .env file if present
load_dotenv()


def env_str(var_name: str, default: str | None = None) -> str | None:
    value = os.environ.get(var_name)
    if value is None or value == "":
        return default
    return value


def env_int(var_name: str, default: int) -> int:
    value = os.environ.get(var_name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        print(
            f"WARNING: Environment variable {var_name} must be an integer. Using default {default}.",
            file=sys.stderr,
        )
        return default


def env_float(var_name: str, default: float) -> float:
    value = os.environ.get(var_name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        print(
            f"WARNING: Environment variable {var_name} must be a float. Using default {default}.",
            file=sys.stderr,
        )
        return default


def _detect_embeddings_provider() -> str:
    provider = (env_str("RAG_EMBEDDINGS_PROVIDER") or "").strip().lower()
    if provider in ("openai", "azure", "google"):
        return provider
    if env_str("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if env_str("GOOGLE_API_KEY"):
        return "google"
    return "openai"


def _detect_chat_provider() -> str:
    provider = (env_str("RAG_CHAT_PROVIDER") or "").strip().lower()
    if provider in ("openai", "azure", "google"):
        return provider
    if env_str("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if env_str("GOOGLE_API_KEY"):
        return "google"
    return "openai"


def require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        print(f"ERROR: Environment variable {var_name} is not set.", file=sys.stderr)
        sys.exit(2)
    return value


def load_markdown_documents(source_dir: str) -> List[Document]:
    if not os.path.isdir(source_dir):
        print(f"ERROR: --source-dir does not exist or is not a directory: {source_dir}", file=sys.stderr)
        sys.exit(2)

    loader = DirectoryLoader(
        source_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True,
        silent_errors=True,
    )
    docs = loader.load()
    return docs


def split_documents(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def _ensure_keys_for_provider(provider: str) -> None:
    if provider == "azure":
        require_env("AZURE_OPENAI_ENDPOINT")
        require_env("AZURE_OPENAI_API_KEY")
        # Accept either AZURE_OPENAI_API_VERSION or OPENAI_API_VERSION
        if not (env_str("AZURE_OPENAI_API_VERSION") or env_str("OPENAI_API_VERSION")):
            print(
                "ERROR: Set either AZURE_OPENAI_API_VERSION or OPENAI_API_VERSION for Azure.",
                file=sys.stderr,
            )
            sys.exit(2)
    elif provider == "google":
        require_env("GOOGLE_API_KEY")
    else:
        require_env("OPENAI_API_KEY")


def _make_embeddings(provider: str, embedding_model: str):
    if provider == "azure":
        azure_deployment = env_str("RAG_AZURE_EMBEDDING_DEPLOYMENT", embedding_model)
        api_version = env_str("AZURE_OPENAI_API_VERSION") or env_str("OPENAI_API_VERSION")
        return AzureOpenAIEmbeddings(azure_deployment=azure_deployment, api_version=api_version)
    if provider == "google":
        google_model = env_str("RAG_GOOGLE_EMBEDDING_MODEL", embedding_model or "models/text-embedding-004")
        # Google Embeddings API expects model names like "models/text-embedding-004"
        if google_model and not google_model.startswith("models/"):
            google_model = f"models/{google_model}"
        api_version = env_str("GOOGLE_API_VERSION", "v1beta")
        return GoogleGenerativeAIEmbeddings(model=google_model, google_api_version=api_version)
    return OpenAIEmbeddings(model=embedding_model)


def _make_chat_llm(provider: str, model: str, temperature: float):
    if provider == "azure":
        deployment = env_str("RAG_AZURE_CHAT_DEPLOYMENT", model)
        api_version = env_str("AZURE_OPENAI_API_VERSION") or env_str("OPENAI_API_VERSION")
        # Many Azure deployments only support default temperature; force default 1.0
        return AzureChatOpenAI(azure_deployment=deployment, api_version=api_version, temperature=1)
    if provider == "google":
        google_model = model or env_str("RAG_GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
        api_version = env_str("GOOGLE_API_VERSION", "v1beta")
        return ChatGoogleGenerativeAI(model=google_model, temperature=temperature, google_api_version=api_version)
    # OpenAI default
    # Some OpenAI models only allow default temperature; force default 1.0
    return ChatOpenAI(model=model, temperature=1)


def index_command(args: argparse.Namespace) -> None:
    emb_provider = _detect_embeddings_provider()
    _ensure_keys_for_provider(emb_provider)

    print("Loading markdown documents...")
    docs = load_markdown_documents(args.source_dir)
    if not docs:
        print("No markdown files found.")
        return

    print(f"Loaded {len(docs)} documents. Splitting into chunks...")
    splits = split_documents(docs, args.chunk_size, args.chunk_overlap)
    total_chunks = len(splits)

    provider = emb_provider
    if emb_provider == "azure":
        embed_name = env_str("RAG_AZURE_EMBEDDING_DEPLOYMENT", args.embedding_model)
    elif emb_provider == "google":
        embed_name = env_str("RAG_GOOGLE_EMBEDDING_MODEL", args.embedding_model or "models/text-embedding-004")
    else:
        embed_name = args.embedding_model
    print(
        f"Embeddings provider: {provider} | Embeddings model: {embed_name} | Collection: {args.collection_name} | Persist dir: {args.persist_dir}"
    )

    embeddings = _make_embeddings(emb_provider, args.embedding_model)

    os.makedirs(args.persist_dir, exist_ok=True)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
        collection_name=args.collection_name,
    )

    batch_size = args.index_batch_size
    print(f"Embedding and indexing {total_chunks} chunks in batches of {batch_size}...")
    indexed = 0
    for start in range(0, total_chunks, batch_size):
        end = min(start + batch_size, total_chunks)
        batch = splits[start:end]
        vector_store.add_documents(batch)
        indexed = end
        percent = int(indexed * 100 / total_chunks) if total_chunks else 100
        print(f"[{percent}%] Indexed {indexed}/{total_chunks}")

    vector_store.persist()
    print(f"Indexing complete. Persisted to: {args.persist_dir} (collection: {args.collection_name})")


def _format_docs(docs: Iterable[Document]) -> str:
    lines: List[str] = []
    for d in docs:
        source = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        lines.append(f"[Source: {os.path.basename(source)}]\n{d.page_content}")
    return "\n\n".join(lines)


def _collect_sources(docs: Iterable[Document]) -> List[str]:
    unique: Set[str] = set()
    result: List[str] = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("path") or "unknown"
        if src not in unique:
            unique.add(src)
            result.append(src)
    return result


def build_rag_chain(vector_store: Chroma, chat_provider: str, model: str, temperature: float):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """
You are a careful assistant. Answer the user question using ONLY the context.
If the answer isn't in the context, say you don't know.

Context:
{context}

Question: {question}

Provide a concise, direct answer. If applicable, include short citations like [filename].
""".strip()
    )

    llm = _make_chat_llm(provider=chat_provider, model=model, temperature=temperature)
    format_docs = RunnableLambda(_format_docs)

    chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | prompt | llm | StrOutputParser()

    return chain, retriever


def query_command(args: argparse.Namespace) -> None:
    emb_provider = _detect_embeddings_provider()
    chat_provider = _detect_chat_provider()
    _ensure_keys_for_provider(emb_provider)
    _ensure_keys_for_provider(chat_provider)

    if not os.path.isdir(args.persist_dir):
        print(
            f"ERROR: --persist-dir does not exist or is not a directory: {args.persist_dir}",
            file=sys.stderr,
        )
        sys.exit(2)

    if chat_provider == "azure":
        chat_name = env_str("RAG_AZURE_CHAT_DEPLOYMENT", args.model)
    elif chat_provider == "google":
        chat_name = env_str("RAG_GOOGLE_CHAT_MODEL", args.model)
    else:
        chat_name = args.model
    if chat_provider == "google":
        api_version_str = env_str("GOOGLE_API_VERSION", "v1beta")
    elif chat_provider == "azure":
        api_version_str = env_str("AZURE_OPENAI_API_VERSION") or env_str("OPENAI_API_VERSION") or "-"
    else:
        api_version_str = "-"
    print(
        f"Embeddings provider: {emb_provider} | Chat provider: {chat_provider} | Chat model: {chat_name} | API version: {api_version_str} | Collection: {args.collection_name}"
    )

    embeddings = _make_embeddings(emb_provider, args.embedding_model)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=args.persist_dir,
        collection_name=args.collection_name,
    )

    chain, retriever = build_rag_chain(
        vector_store=vector_store, chat_provider=chat_provider, model=args.model, temperature=args.temperature
    )

    question = args.question
    if not question:
        print("ERROR: --question is required for query command", file=sys.stderr)
        sys.exit(2)

    # Retrieve docs explicitly to display sources
    print("Retrieving context (k=4)...")
    retrieved_docs = retriever.invoke(question)
    print(f"Retrieved {len(retrieved_docs)} documents. Generating answer...")
    answer = chain.invoke(question)

    print("\n=== Answer ===\n")
    print(answer)

    sources = _collect_sources(retrieved_docs)
    if sources:
        print("\n=== Sources ===")
        for s in sources:
            print(f"- {s}")


def build_parser() -> argparse.ArgumentParser:
    default_repo_root = os.path.dirname(os.path.abspath(__file__))
    default_source = os.path.join(default_repo_root, "Consumer Platforms.wiki")
    default_persist = os.path.join(default_repo_root, ".chroma")

    parser = argparse.ArgumentParser(description="RAG CLI over markdown wiki with Chroma")
    subparsers = parser.add_subparsers(dest="command", required=True)

    is_azure = bool(env_str("AZURE_OPENAI_ENDPOINT"))
    is_google = bool(env_str("GOOGLE_API_KEY"))

    # index
    p_index = subparsers.add_parser("index", help="Index markdown files into Chroma")
    p_index.add_argument(
        "--source-dir",
        type=str,
        default=env_str("RAG_SOURCE_DIR", default_source),
        help="Directory containing .md files (default: '<repo>/Consumer Platforms.wiki')",
    )
    p_index.add_argument(
        "--persist-dir",
        type=str,
        default=env_str("RAG_PERSIST_DIR", default_persist),
        help="Directory to persist Chroma data (default: '<repo>/.chroma')",
    )
    p_index.add_argument(
        "--collection-name",
        type=str,
        default=env_str("RAG_COLLECTION_NAME", "wiki-md"),
        help="Chroma collection name",
    )
    p_index.add_argument(
        "--chunk-size",
        type=int,
        default=env_int("RAG_CHUNK_SIZE", 1000),
        help="Chunk size",
    )
    p_index.add_argument(
        "--chunk-overlap",
        type=int,
        default=env_int("RAG_CHUNK_OVERLAP", 200),
        help="Chunk overlap",
    )
    p_index.add_argument(
        "--embedding-model",
        type=str,
        default=(
            env_str("RAG_GOOGLE_EMBEDDING_MODEL", "text-embedding-004") if is_google
            else env_str("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        ),
        help="OpenAI embeddings model",
    )
    p_index.add_argument(
        "--index-batch-size",
        type=int,
        default=env_int("RAG_INDEX_BATCH_SIZE", 128),
        help="Number of chunks per embedding/indexing batch",
    )
    p_index.set_defaults(func=index_command)

    # query
    p_query = subparsers.add_parser("query", help="Query the RAG index and generate an answer")
    p_query.add_argument(
        "--question",
        type=str,
        required=False,
        default=env_str("RAG_QUESTION", None),
        help="Question to ask (or set RAG_QUESTION)",
    )
    p_query.add_argument(
        "--persist-dir",
        type=str,
        default=env_str("RAG_PERSIST_DIR", default_persist),
        help="Directory where Chroma data is persisted (default: '<repo>/.chroma')",
    )
    p_query.add_argument(
        "--collection-name",
        type=str,
        default=env_str("RAG_COLLECTION_NAME", "wiki-md"),
        help="Chroma collection name",
    )
    p_query.add_argument(
        "--model",
        type=str,
        default=(
            env_str("RAG_GOOGLE_CHAT_MODEL", "gemini-2.5-flash") if is_google
            else env_str("RAG_CHAT_MODEL", "gpt-4o-mini")
        ),
        help="OpenAI chat model name",
    )
    p_query.add_argument(
        "--temperature",
        type=float,
        default=env_float("RAG_TEMPERATURE", 0.0),
        help="Sampling temperature for the chat model",
    )
    p_query.add_argument(
        "--embedding-model",
        type=str,
        default=env_str("RAG_EMBEDDING_MODEL", "text-embedding-3-small"),
        help="OpenAI embeddings model (must match the one used at index time for best results)",
    )
    p_query.set_defaults(func=query_command)

    return parser


def main(argv: List[str]) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])


