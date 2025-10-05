## RAG over Markdown Wiki (LangChain + Chroma)

This project provides a small CLI to build a Retrieval-Augmented Generation (RAG) index from local Markdown files and query it using OpenAI (or Azure OpenAI) chat models with Chroma as the vector store.

Inspired by the LangChain RAG tutorial: [`https://python.langchain.com/docs/tutorials/rag/`].

### Features
- Index all `.md` files recursively from a directory.
- Persist embeddings to a local Chroma database.
- Query with context-constrained answers and file citations.
- Works with OpenAI, Azure OpenAI, or Google Gemini (auto-detected via env).

### Requirements
- Python 3.10+
- An OpenAI API key or an Azure OpenAI resource and API key

### Installation
```bash
cd /Users/jmffraiz/BAT-Repos/RAG
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration (.env)
Copy the example and edit values:
```bash
cp env.example .env
```

Key variables:
- `OPENAI_API_KEY`: Required for OpenAI (ignored if Azure endpoint is set)
- `RAG_SOURCE_DIR`: Folder with your wiki Markdown files
- `RAG_PERSIST_DIR`: Folder where Chroma persists the index
- `RAG_COLLECTION_NAME`: Chroma collection name
- `RAG_EMBEDDINGS_PROVIDER`, `RAG_CHAT_PROVIDER`: Choose per-provider (openai | azure | google)
- `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`: Text splitting parameters
- `RAG_EMBEDDING_MODEL`: Embedding model (OpenAI or Azure deployment name)
- `RAG_CHAT_MODEL`, `RAG_TEMPERATURE`: Chat settings
- Optional `RAG_QUESTION`: Default question for quick tests

#### Azure OpenAI
If these are set, the CLI switches to Azure automatically:
- `AZURE_OPENAI_ENDPOINT` (e.g. `https://lh-poc-azure-openai.openai.azure.com/`)
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION` (e.g. `2024-02-15-preview`)
- `RAG_AZURE_CHAT_DEPLOYMENT` (e.g. `gpt-5`)
- `RAG_AZURE_EMBEDDING_DEPLOYMENT` (e.g. `text-embedding-3-small`)

#### Google Gemini
If `GOOGLE_API_KEY` is set, the CLI uses Gemini via `langchain-google-genai`:
- `GOOGLE_API_KEY`
- Optional overrides:
  - `RAG_GOOGLE_CHAT_MODEL` (default `gemini-2.5-flash`)
  - `RAG_GOOGLE_EMBEDDING_MODEL` (default `models/text-embedding-004`)
  - `GOOGLE_API_VERSION` (default `v1beta`; try `v1beta` if models 404 on `v1`)

### Usage

Index your wiki into Chroma:
```bash
python rag_cli.py index
```

Query the index:
```bash
python rag_cli.py query --question "What is the AEM Customer Reviews component for?"
```

Both commands read defaults from `.env`. You can override via flags (`--source-dir`, `--persist-dir`, `--model`, etc.).

### File structure
- `rag_cli.py`: CLI entry point (index and query subcommands)
- `requirements.txt`: Python dependencies
- `env.example`: Example environment configuration
- `.chroma/`: Default Chroma persistence directory (created at runtime)

### Notes
- Only `.md` files are loaded.
- Indexing can be re-run to refresh the persisted collection.
- Token/throughput limits (e.g., Azure 250k TPM) are enforced by the provider.

### Troubleshooting
- Import warnings in your editor usually clear after installing requirements and activating the venv.
- If `OPENAI_API_KEY` is missing (and Azure endpoint isn’t set), the script will exit with an error.
- Ensure the `RAG_SOURCE_DIR` exists and contains Markdown files.

### MCP Server (Cursor / Model Context Protocol)

This repo includes an MCP server exposing the RAG query as a tool.

- Entry point: `mcp_rag_server.py`
- Tool: `query`
  - Args: `question` (required), `persist_dir`, `collection_name`, `model`, `temperature`, `embedding_model`

Run locally (stdio):
```bash
python mcp_rag_server.py
```

Cursor MCP client config example (macOS): add to `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "rag": {
      "command": "/Users/jmffraiz/BAT-Repos/RAG/.venv/bin/python",
      "args": ["/Users/jmffraiz/BAT-Repos/RAG/mcp_rag_server.py"],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

Use the tool in Cursor:
- Open the MCP Tools panel, select `rag`, run `query` with your `question`.
- Ensure `.env` is configured (see Configuration section); the server auto-detects OpenAI/Azure/Google providers.


