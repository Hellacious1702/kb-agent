# KB Agent - Internal Knowledge Base Agent

A RAG-powered document Q&A system for internal knowledge management. Upload documents (PDF, DOCX, TXT, MD), ask questions, and get cited answers with source tracking.

## Features

- **Document Ingestion**: PDF (PyMuPDF), DOCX (python-docx), TXT/MD with unstructured fallback
- **Smart Chunking**: Fixed-size, semantic, and hybrid chunking strategies
- **Embeddings**: Provider-agnostic via LiteLLM (OpenAI, Azure, Cohere, etc.)
- **Hybrid Retrieval**: Vector search (ChromaDB) + BM25 keyword search with weighted fusion
- **RAG Generation**: Streaming answers with inline citations [DocName, Chunk N, Page X]
- **Web UI**: Chat interface with source sidebar, admin panel for document management
- **Multi-tenant**: Workspace isolation with API key authentication
- **Developer Experience**: Docker, Fly.io deployment, GitHub Actions CI/CD

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Documents  │────▶│  Chunker     │────▶│  Embedder   │
│  (PDF,etc)  │     │(Fixed/Semant)│     │  (LiteLLM)  │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                   ┌──────────────┐     ┌───────▼───────┐
                   │  ChromaDB    │◀────│  Vector Store │
                   │  (Vectors)   │     └───────────────┘
                   └──────┬───────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
    ┌─────────┐      ┌──────────┐     ┌─────────┐
    │  BM25   │      │ Retriever │     │ Generator│
    │  Index  │─────▶│ (Hybrid)  │────▶│ (RAG)   │
    └─────────┘      └──────────┘     └────┬────┘
                                            │
                                            ▼
                                    ┌───────────────┐
                                    │  Chat UI      │
                                    │  + Citations  │
                                    └───────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- ChromaDB
- OpenAI API key (or compatible provider)

### Local Development

```bash
# Clone and setup
cd kb-agent
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services (PostgreSQL + ChromaDB)
docker-compose up -d postgres chromadb

# Run migrations
alembic upgrade head

# Start app
uvicorn app.main:app --reload
```

Open http://localhost:8000 for chat, http://localhost:8000/admin for admin panel.

### Docker

```bash
# Build and run
docker-compose up --build

# Or build image
docker build -t kb-agent .
docker run -p 8000:8000 --env-file .env kb-agent
```

### Fly.io Deployment

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Login and launch
fly auth login
fly launch

# Set secrets
fly secrets set LITELLM_API_KEY=sk-... SECRET_KEY=your-secret

# Deploy
fly deploy
```

## Configuration

Key environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/kb_agent` |
| `CHROMA_HOST` | ChromaDB host | `localhost` |
| `CHROMA_PORT` | ChromaDB port | `8000` |
| `LITELLM_API_KEY` | LLM provider API key | Required |
| `LITELLM_MODEL` | Generation model | `gpt-4o-mini` |
| `LITELLM_EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `SECRET_KEY` | JWT secret | Required |
| `CHUNK_SIZE` | Chunk size in chars | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap in chars | `200` |

## API Endpoints

### Workspaces
- `POST /workspaces` - Create workspace
- `GET /workspaces` - List workspaces
- `GET /workspaces/{id}` - Get workspace
- `PATCH /workspaces/{id}` - Update workspace
- `DELETE /workspaces/{id}` - Delete workspace

### API Keys
- `POST /workspaces/{id}/api-keys` - Create API key
- `GET /workspaces/{id}/api-keys` - List API keys
- `DELETE /workspaces/{id}/api-keys/{key_id}` - Revoke API key

### Documents
- `POST /documents/upload` - Upload document
- `GET /documents` - List documents
- `GET /documents/{id}` - Get document
- `DELETE /documents/{id}` - Delete document

### Chat
- `POST /chat` - Query knowledge base
- `POST /chat/stream` - Stream query response
- `GET /chat/history` - Get query history

### Admin
- `GET /workspaces/{id}/stats` - Workspace statistics
- `GET /workspaces/{id}/chunks` - List chunks
- `POST /workspaces/{id}/reindex` - Rebuild search index
- `GET /workspaces/{id}/documents` - List documents with chunk counts

## Authentication

Two authentication methods:

1. **API Key** (for programmatic access):
   ```
   X-API-Key: sk_<prefix>_<random>
   ```

2. **JWT Token** (for web UI):
   ```
   Authorization: Bearer <token>
   ```

API keys are workspace-scoped and hashed with bcrypt.

## Chunking Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `FixedSizeChunker` | Fixed chars with overlap, sentence-boundary aware | General purpose, predictable sizes |
| `SemanticChunker` | Groups sentences by embedding similarity | Complex documents, topic coherence |
| `HybridChunker` | Semantic first, fixed-size fallback for large chunks | Best of both worlds (default) |

## Hybrid Retrieval

Combines vector similarity (cosine) and BM25 keyword scores:

```
fused_score = vector_weight * normalized_vector_score + bm25_weight * normalized_bm25_score
```

Default weights: `vector_weight=0.7`, `bm25_weight=0.3`

## Citation Format

Inline citations in answers: `[DocumentName, Chunk N, Page X]`

Source sidebar shows:
- Document filename
- Chunk index
- Page number
- Relevance score
- Source type (vector/bm25/hybrid)
- Content preview

## Project Structure

```
kb-agent/
├── app/
│   ├── api/           # FastAPI routes
│   │   ├── admin.py   # Admin endpoints
│   │   ├── chat.py    # Chat/RAG endpoints
│   │   ├── deps.py    # Dependencies
│   │   ├── documents.py # Document endpoints
│   │   └── workspaces.py # Workspace/API key endpoints
│   ├── core/
│   │   ├── config.py  # Settings
│   │   └── database.py # DB session
│   ├── models/        # SQLAlchemy models
│   ├── services/
│   │   ├── chunker.py      # Chunking strategies
│   │   ├── document_processor.py # PDF/DOCX/TXT extraction
│   │   ├── embedder.py     # LiteLLM embeddings
│   │   ├── rag_generator.py # RAG with citations
│   │   └── retriever.py    # Hybrid retrieval
│   ├── utils/
│   │   └── auth.py    # API key/JWT auth
│   ├── web/
│   │   ├── templates/ # Jinja2 templates
│   │   └── static/    # Static assets
│   └── main.py        # FastAPI app
├── tests/             # Pytest tests
├── docker/            # Docker configs
├── .github/workflows/ # CI/CD
├── Dockerfile
├── docker-compose.yml
├── fly.toml
├── pyproject.toml
└── README.md
```

## Development

```bash
# Run tests
pytest -v

# Lint
ruff check .
ruff format .

# Type check
mypy app/

# Pre-commit
pre-commit install
```

## License

MIT