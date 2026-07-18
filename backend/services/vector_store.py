import numpy as np
# Hot-patch for ChromaDB compatibility with NumPy 2.0+
np.float_ = np.float64

import os
import logging
from datetime import datetime
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

logger = logging.getLogger("sera.vector_store")

class APEXEmbeddingFunction(EmbeddingFunction):
    """
    ChromaDB-compatible embedding function that reuses the pre-loaded APEX
    SentenceTransformer encoder for resource efficiency and offline speed.
    """
    def __call__(self, input: Documents) -> Embeddings:
        try:
            from entity_interface.apex_causal import get_encoder
            encoder = get_encoder()
            embeddings = encoder.encode(input, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate embeddings using APEX encoder: {e}")
            # Fallback to zero vectors in case of emergency
            return [[0.0] * 384 for _ in range(len(input))]

class VectorStoreService:
    _client = None
    _collection = None

    @classmethod
    def _get_collection(cls):
        if cls._collection is not None:
            return cls._collection

        try:
            # Resolve directory for Chroma storage inside backend/data/
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            chroma_dir = os.path.join(base_dir, "data", "chroma_db")
            os.makedirs(chroma_dir, exist_ok=True)
            
            cls._client = chromadb.PersistentClient(path=chroma_dir)
            cls._collection = cls._client.get_or_create_collection(
                name="sera_knowledge_base",
                embedding_function=APEXEmbeddingFunction()
            )
            logger.info("ChromaDB persistent client initialized successfully.")
            return cls._collection
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB collection: {e}")
            return None

    @classmethod
    def add_document(cls, doc_id: str, text: str, metadata: dict = None):
        """Indexes a raw text chunk into ChromaDB."""
        collection = cls._get_collection()
        if collection is None:
            logger.warning("ChromaDB collection is offline. Skipping indexing.")
            return
            
        try:
            # Use upsert to avoid duplicate keys
            collection.upsert(
                ids=[doc_id],
                documents=[text],
                metadatas=[metadata or {}]
            )
            logger.debug(f"Document {doc_id} successfully indexed in vector store.")
        except Exception as e:
            logger.error(f"Failed to index document {doc_id}: {e}")

    @classmethod
    def index_news(cls, gdelt_id: str, title: str, themes: str, tone: float, date: datetime):
        text = f"NEWS ARTICLE | Title: {title} | Themes: {themes or 'General'} | Sentiment Tone: {tone:.2f} | Date: {date.isoformat()}"
        metadata = {
            "type": "news",
            "gdelt_id": gdelt_id,
            "date": date.isoformat()
        }
        cls.add_document(f"news_{gdelt_id}", text, metadata)

    @classmethod
    def index_filing(cls, ticker: str, revenue: float, accounts_receivable: float, deferred_revenue: float, date: datetime):
        text = f"SEC FINANCIAL FILING | Company Ticker: {ticker} | Quarterly Revenue: ${revenue:,.2f} | Accounts Receivable: ${accounts_receivable:,.2f} | Deferred Revenue: ${deferred_revenue:,.2f} | Date: {date.isoformat()}"
        metadata = {
            "type": "filing",
            "ticker": ticker,
            "date": date.isoformat()
        }
        cls.add_document(f"filing_{ticker}_{date.strftime('%Y%m%d')}", text, metadata)

    @classmethod
    def index_executive_movement(cls, ticker: str, exec_name: str, old_title: str, new_title: str, change_type: str, date: datetime):
        old_lbl = old_title or "None (External)"
        new_lbl = new_title or "Role Exited"
        text = f"EXECUTIVE LEADERSHIP TRANSITION | Company Ticker: {ticker} | Executive Name: {exec_name} | Old Role: {old_lbl} | New Role: {new_lbl} | Transition Type: {change_type.upper()} | Date: {date.isoformat()}"
        metadata = {
            "type": "executive",
            "ticker": ticker,
            "exec_name": exec_name,
            "change_type": change_type,
            "date": date.isoformat()
        }
        cls.add_document(f"exec_{ticker}_{exec_name}_{date.strftime('%Y%m%d')}", text, metadata)

    @classmethod
    def index_healthcare_metric(cls, region: str, admissions: int, avg_payment: float, drug_claims: int, date: datetime):
        text = f"REGIONAL HEALTHCARE TELEMETRY | Indian State Code: {region} | Hospital Admissions Count: {admissions:,} | Avg Patient Treatment Cost: INR {avg_payment:,.2f} | Drug Claims Volume: {drug_claims:,} | Date: {date.isoformat()}"
        metadata = {
            "type": "healthcare",
            "region": region,
            "date": date.isoformat()
        }
        cls.add_document(f"healthcare_{region}_{date.strftime('%Y%m%d')}", text, metadata)

    @classmethod
    def query_knowledge_base(cls, query_text: str, limit: int = 5) -> list[str]:
        """Queries ChromaDB collection for top matching document texts."""
        collection = cls._get_collection()
        if collection is None:
            logger.warning("ChromaDB collection is offline. Returning empty context.")
            return []
            
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=limit
            )
            documents = results.get("documents", [[]])
            return documents[0] if documents else []
        except Exception as e:
            logger.error(f"Failed to query knowledge base: {e}")
            return []
