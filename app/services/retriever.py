"""Hybrid retrieval combining vector search (ChromaDB) and BM25."""

import logging
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.services.embedder import embedder

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class RetrievalResult:
    """Result from hybrid retrieval."""
    chunk_id: str
    document_id: str
    content: str
    page_number: int | None
    char_start: int
    char_end: int
    score: float
    metadata: dict
    source_type: str  # "vector", "bm25", or "hybrid"


class VectorStore:
    """ChromaDB vector store wrapper."""

    def __init__(self):
        self._client: chromadb.AsyncHttpClient | None = None
        self._collection = None

    async def initialize(self):
        """Initialize ChromaDB connection and collection."""
        self._client = chromadb.AsyncHttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = await self._client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Connected to ChromaDB collection: {settings.chroma_collection_name}")

    async def close(self):
        """Close ChromaDB connection."""
        if self._client:
            await self._client.close()

    async def upsert_chunks(
        self,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """Upsert chunks to vector store."""
        if not self._collection:
            await self.initialize()

        await self._collection.upsert(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug(f"Upserted {len(chunk_ids)} chunks to vector store")

    async def query(
        self,
        query_embedding: list[float],
        workspace_id: str,
        n_results: int = 10,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        """Query vector store for similar chunks."""
        if not self._collection:
            await self.initialize()

        # Build where clause
        where_clause = {"workspace_id": workspace_id}
        if where:
            where_clause.update(where)

        results = await self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        retrieval_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert cosine distance to similarity score (1 - distance)
                score = 1 - distance

                retrieval_results.append(RetrievalResult(
                    chunk_id=chunk_id,
                    document_id=results["metadatas"][0][i].get("document_id", ""),
                    content=results["documents"][0][i],
                    page_number=results["metadatas"][0][i].get("page_number"),
                    char_start=results["metadatas"][0][i].get("char_start", 0),
                    char_end=results["metadatas"][0][i].get("char_end", 0),
                    score=score,
                    metadata=results["metadatas"][0][i],
                    source_type="vector",
                ))

        return retrieval_results

    async def delete_workspace_chunks(self, workspace_id: str):
        """Delete all chunks for a workspace."""
        if not self._collection:
            await self.initialize()

        await self._collection.delete(where={"workspace_id": workspace_id})
        logger.info(f"Deleted all chunks for workspace {workspace_id}")

    async def get_all_chunks_for_workspace(self, workspace_id: str) -> list[dict]:
        """Get all chunks for a workspace (for BM25 indexing)."""
        if not self._collection:
            await self.initialize()

        results = await self._collection.get(
            where={"workspace_id": workspace_id},
            include=["documents", "metadatas"],
        )

        chunks = []
        if results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": results["metadatas"][i].get("document_id", ""),
                    "content": results["documents"][i],
                    "page_number": results["metadatas"][i].get("page_number"),
                    "char_start": results["metadatas"][i].get("char_start", 0),
                    "char_end": results["metadatas"][i].get("char_end", 0),
                    "metadata": results["metadatas"][i],
                })
        return chunks


class BM25Index:
    """BM25 keyword search index."""

    def __init__(self):
        self._index: BM25Okapi | None = None
        self._chunk_ids: list[str] = []
        self._chunk_data: list[dict] = []

    def build(self, chunks: list[dict]):
        """Build BM25 index from chunks."""
        # Tokenize documents (simple whitespace + punctuation splitting)
        tokenized_docs = []
        self._chunk_ids = []
        self._chunk_data = []

        for chunk in chunks:
            # Simple tokenization - can be improved with proper tokenizer
            tokens = self._tokenize(chunk["content"])
            tokenized_docs.append(tokens)
            self._chunk_ids.append(chunk["chunk_id"])
            self._chunk_data.append(chunk)

        self._index = BM25Okapi(tokenized_docs)
        logger.info(f"Built BM25 index with {len(chunks)} documents")

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        import re
        # Lowercase, split on non-alphanumeric
        return re.findall(r'\b\w+\b', text.lower())

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search BM25 index."""
        if not self._index:
            return []

        query_tokens = self._tokenize(query)
        scores = self._index.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = self._chunk_data[idx]
            results.append(RetrievalResult(
                chunk_id=chunk["chunk_id"],
                document_id=chunk["document_id"],
                content=chunk["content"],
                page_number=chunk["page_number"],
                char_start=chunk["char_start"],
                char_end=chunk["char_end"],
                score=float(scores[idx]),
                metadata=chunk["metadata"],
                source_type="bm25",
            ))

        return results


class HybridRetriever:
    """
    Hybrid retriever combining vector similarity and BM25 keyword search.
    
    WHY: Vector search excels at semantic similarity but misses exact keyword
    matches. BM25 excels at exact keyword matching but misses semantic
    relationships. Combining both (with reciprocal rank fusion or weighted
    scoring) provides better recall and precision than either alone.
    
    Strategy: Retrieve from both, then fuse scores using weighted combination
    (configurable vector_weight + bm25_weight = 1.0).
    """

    def __init__(
        self,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        top_k: int = 10,
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self._initialized = False

    async def initialize(self):
        """Initialize both retrievers."""
        await self.vector_store.initialize()
        # Build BM25 index from existing chunks
        await self._rebuild_bm25()
        self._initialized = True

    async def _rebuild_bm25(self):
        """Rebuild BM25 index from vector store."""
        # This would need a workspace_id - for now, we'll rebuild per workspace
        pass

    async def rebuild_bm25_for_workspace(self, workspace_id: str):
        """Rebuild BM25 index for a specific workspace."""
        chunks = await self.vector_store.get_all_chunks_for_workspace(workspace_id)
        self.bm25_index.build(chunks)

    async def add_chunks(
        self,
        workspace_id: str,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ):
        """Add chunks to both vector store and BM25 index."""
        await self.vector_store.upsert_chunks(chunk_ids, embeddings, documents, metadatas)
        # Rebuild BM25 for this workspace
        await self.rebuild_bm25_for_workspace(workspace_id)

    async def retrieve(
        self,
        query: str,
        workspace_id: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval: combine vector and BM25 results.
        
        WHY: Reciprocal Rank Fusion (RRF) is a parameter-free fusion method
        that works well in practice. Alternative: weighted score combination.
        We use weighted combination here for interpretability.
        """
        if not self._initialized:
            await self.initialize()

        k = top_k or self.top_k

        # Get query embedding
        query_embedding = await embedder.embed_single(query)

        # Retrieve from both sources (get more for better fusion)
        fetch_k = k * 3
        vector_results = await self.vector_store.query(
            query_embedding, workspace_id, n_results=fetch_k
        )
        bm25_results = self.bm25_index.search(query, top_k=fetch_k)

        # Fuse results
        fused = self._fuse_results(vector_results, bm25_results, k)
        return fused

    def _fuse_results(
        self,
        vector_results: list[RetrievalResult],
        bm25_results: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Fuse vector and BM25 results using weighted score combination."""
        # Create maps for deduplication
        chunk_scores: dict[str, dict] = {}

        # Normalize scores to [0, 1] range
        vector_scores = [r.score for r in vector_results]
        bm25_scores = [r.score for r in bm25_results]

        max_vec = max(vector_scores) if vector_scores else 1.0
        max_bm25 = max(bm25_scores) if bm25_scores else 1.0

        # Process vector results
        for r in vector_results:
            norm_score = r.score / max_vec if max_vec > 0 else 0
            chunk_scores[r.chunk_id] = {
                "result": r,
                "vector_score": norm_score,
                "bm25_score": 0.0,
            }

        # Process BM25 results
        for r in bm25_results:
            norm_score = r.score / max_bm25 if max_bm25 > 0 else 0
            if r.chunk_id in chunk_scores:
                chunk_scores[r.chunk_id]["bm25_score"] = norm_score
                chunk_scores[r.chunk_id]["result"].source_type = "hybrid"
            else:
                chunk_scores[r.chunk_id] = {
                    "result": r,
                    "vector_score": 0.0,
                    "bm25_score": norm_score,
                }

        # Calculate fused scores
        fused_results = []
        for chunk_id, data in chunk_scores.items():
            fused_score = (
                self.vector_weight * data["vector_score"] +
                self.bm25_weight * data["bm25_score"]
            )
            result = data["result"]
            result.score = fused_score
            fused_results.append(result)

        # Sort by fused score and return top-k
        fused_results.sort(key=lambda r: r.score, reverse=True)
        return fused_results[:top_k]

    async def close(self):
        """Close connections."""
        await self.vector_store.close()


# Global retriever instance
hybrid_retriever = HybridRetriever()
