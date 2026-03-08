"""
Vector Memory Store
向量记忆存储

Long-term memory storage using vector databases with:
- FAISS for fast similarity search
- ChromaDB for metadata storage
- Redis for caching
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import hashlib
import os

# Try importing optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class MemoryEntry:
    """Memory entry with metadata"""
    id: str
    vector: np.ndarray
    content: str
    memory_type: str  # "fact", "experience", "concept", "skill"
    timestamp: datetime
    source: str = ""
    confidence: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "confidence": self.confidence,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], vector: np.ndarray) -> "MemoryEntry":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            vector=vector,
            content=data["content"],
            memory_type=data["memory_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source=data.get("source", ""),
            confidence=data.get("confidence", 1.0),
            access_count=data.get("access_count", 0),
            last_accessed=datetime.fromisoformat(data["last_accessed"]) if data.get("last_accessed") else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class FAISSMemoryStore:
    """
    FAISS-based vector memory store
    基于FAISS的向量记忆存储
    """
    
    def __init__(
        self,
        dim: int,
        index_type: str = "IndexFlatIP",  # Inner product (cosine similarity for normalized vectors)
        nlist: int = 100,  # For IVF indexes
        nprobe: int = 10,
    ):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not installed. Install with: pip install faiss-cpu")
        
        self.dim = dim
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe
        
        # Create index
        self.index = self._create_index()
        
        # ID mapping (FAISS uses internal IDs)
        self.id_map: Dict[int, str] = {}
        self.reverse_id_map: Dict[str, int] = {}
        self.next_id = 0
        
        # Metadata storage
        self.metadata: Dict[str, MemoryEntry] = {}
        
    def _create_index(self):
        """Create FAISS index"""
        if self.index_type == "IndexFlatIP":
            return faiss.IndexFlatIP(self.dim)
        elif self.index_type == "IndexFlatL2":
            return faiss.IndexFlatL2(self.dim)
        elif self.index_type == "IndexIVFFlat":
            quantizer = faiss.IndexFlatIP(self.dim)
            return faiss.IndexIVFFlat(quantizer, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        elif self.index_type == "IndexHNSW":
            index = faiss.IndexHNSWFlat(self.dim, 32)
            index.hnsw.efConstruction = 200
            return index
        else:
            return faiss.IndexFlatIP(self.dim)
    
    def add(self, entry: MemoryEntry) -> str:
        """Add memory entry"""
        # Normalize vector for cosine similarity
        vector = entry.vector.astype(np.float32)
        vector = vector / np.linalg.norm(vector)
        
        # Add to index
        faiss_id = self.next_id
        self.index.add(np.expand_dims(vector, 0))
        
        # Update ID mapping
        self.id_map[faiss_id] = entry.id
        self.reverse_id_map[entry.id] = faiss_id
        self.next_id += 1
        
        # Store metadata
        self.metadata[entry.id] = entry
        
        return entry.id
    
    def add_batch(self, entries: List[MemoryEntry]) -> List[str]:
        """Add multiple entries"""
        ids = []
        vectors = []
        
        for entry in entries:
            vector = entry.vector.astype(np.float32)
            vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
            
            faiss_id = self.next_id
            self.id_map[faiss_id] = entry.id
            self.reverse_id_map[entry.id] = faiss_id
            self.next_id += 1
            ids.append(entry.id)
            
            self.metadata[entry.id] = entry
        
        # Batch add to index
        if vectors:
            self.index.add(np.stack(vectors))
        
        return ids
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for similar memories
        
        Args:
            query: Query vector
            k: Number of results
            filter_fn: Optional filter function
            
        Returns:
            List of (entry, score) tuples
        """
        if len(self.metadata) == 0:
            return []
        
        # Normalize query
        query = query.astype(np.float32)
        query = query / np.linalg.norm(query)
        
        # Search
        scores, indices = self.index.search(np.expand_dims(query, 0), k * 2)  # Get more for filtering
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            entry_id = self.id_map.get(int(idx))
            if entry_id and entry_id in self.metadata:
                entry = self.metadata[entry_id]
                
                # Apply filter
                if filter_fn is None or filter_fn(entry):
                    # Update access stats
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    results.append((entry, float(score)))
                    
                    if len(results) >= k:
                        break
        
        return results
    
    def delete(self, entry_id: str) -> bool:
        """Delete memory entry"""
        if entry_id not in self.reverse_id_map:
            return False
        
        # Note: FAISS doesn't support deletion, so we mark as deleted
        # For true deletion, need to rebuild index
        if entry_id in self.metadata:
            del self.metadata[entry_id]
            return True
        
        return False
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get memory entry by ID"""
        return self.metadata.get(entry_id)
    
    def update(self, entry_id: str, **kwargs) -> bool:
        """Update memory entry"""
        if entry_id not in self.metadata:
            return False
        
        entry = self.metadata[entry_id]
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
        
        return True
    
    def save(self, path: str):
        """Save index and metadata"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        
        # Save metadata
        metadata_dict = {
            entry_id: {
                "entry": entry.to_dict(),
                "vector": entry.vector.tobytes(),
            }
            for entry_id, entry in self.metadata.items()
        }
        
        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump({
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "next_id": self.next_id,
                "metadata": metadata_dict,
            }, f, default=str)
    
    def load(self, path: str):
        """Load index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        
        # Load metadata
        with open(os.path.join(path, "metadata.json"), "r") as f:
            data = json.load(f)
        
        self.id_map = {int(k): v for k, v in data["id_map"].items()}
        self.reverse_id_map = data["reverse_id_map"]
        self.next_id = data["next_id"]
        
        # Reconstruct metadata
        self.metadata = {}
        for entry_id, entry_data in data["metadata"].items():
            vector = np.frombuffer(entry_data["vector"], dtype=np.float32)
            self.metadata[entry_id] = MemoryEntry.from_dict(entry_data["entry"], vector)
    
    def __len__(self) -> int:
        return len(self.metadata)


class ChromaMemoryStore:
    """
    ChromaDB-based memory store
    基于ChromaDB的记忆存储
    """
    
    def __init__(
        self,
        collection_name: str = "brain_memory",
        persist_directory: str = "./chroma_db",
    ):
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB is not installed. Install with: pip install chromadb")
        
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def add(self, entry: MemoryEntry) -> str:
        """Add memory entry"""
        self.collection.add(
            ids=[entry.id],
            embeddings=[entry.vector.tolist()],
            documents=[entry.content],
            metadatas=[entry.to_dict()],
        )
        return entry.id
    
    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Search for similar memories"""
        results = self.collection.query(
            query_embeddings=[query.tolist()],
            n_results=k,
            where=filter_dict,
        )
        
        entries = []
        for i, (id_, distance, metadata) in enumerate(zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        )):
            # Convert distance to similarity (Chroma uses cosine distance)
            similarity = 1 - distance
            
            # Get embedding
            embedding = results["embeddings"][0][i] if results["embeddings"] else np.zeros(len(query))
            
            entry = MemoryEntry.from_dict(metadata, np.array(embedding))
            entries.append((entry, similarity))
        
        return entries
    
    def delete(self, entry_id: str) -> bool:
        """Delete memory entry"""
        try:
            self.collection.delete(ids=[entry_id])
            return True
        except:
            return False
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get memory entry by ID"""
        try:
            result = self.collection.get(ids=[entry_id], include=["embeddings", "metadatas"])
            if result["ids"]:
                metadata = result["metadatas"][0]
                embedding = np.array(result["embeddings"][0])
                return MemoryEntry.from_dict(metadata, embedding)
        except:
            pass
        return None


class UnifiedMemoryStore:
    """
    Unified memory store combining multiple backends
    统一记忆存储 - 结合多种后端
    """
    
    def __init__(
        self,
        dim: int = 768,
        primary_backend: str = "faiss",
        cache_size: int = 1000,
        persist_dir: str = "./memory_store",
    ):
        self.dim = dim
        self.primary_backend = primary_backend
        self.persist_dir = persist_dir
        
        # Initialize primary store
        if primary_backend == "faiss" and FAISS_AVAILABLE:
            self.primary_store = FAISSMemoryStore(dim)
        elif primary_backend == "chroma" and CHROMA_AVAILABLE:
            self.primary_store = ChromaMemoryStore(persist_directory=persist_dir)
        else:
            # Fallback to simple numpy store
            self.primary_store = None
            self.vectors = []
            self.entries = []
        
        # LRU cache for hot memories
        self.cache: Dict[str, MemoryEntry] = {}
        self.cache_order: deque = deque(maxlen=cache_size)
        
    def add(
        self,
        content: str,
        vector: np.ndarray,
        memory_type: str = "fact",
        source: str = "",
        tags: List[str] = None,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """Add memory"""
        # Generate ID
        entry_id = hashlib.md5(
            f"{content}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        entry = MemoryEntry(
            id=entry_id,
            vector=vector,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        # Add to primary store
        if self.primary_store:
            self.primary_store.add(entry)
        else:
            self.vectors.append(vector)
            self.entries.append(entry)
        
        # Add to cache
        self._add_to_cache(entry)
        
        return entry_id
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_confidence: float = 0.0,
    ) -> List[Tuple[MemoryEntry, float]]:
        """
        Search memories
        
        Args:
            query_vector: Query embedding
            k: Number of results
            memory_type: Filter by memory type
            tags: Filter by tags
            min_confidence: Minimum confidence score
            
        Returns:
            List of (entry, score) tuples
        """
        # Define filter function
        def filter_fn(entry: MemoryEntry) -> bool:
            if memory_type and entry.memory_type != memory_type:
                return False
            if tags and not any(tag in entry.tags for tag in tags):
                return False
            if entry.confidence < min_confidence:
                return False
            return True
        
        # Search primary store
        if self.primary_store:
            results = self.primary_store.search(query_vector, k, filter_fn)
        else:
            # Simple numpy search
            results = self._numpy_search(query_vector, k, filter_fn)
        
        # Update cache with results
        for entry, _ in results:
            self._add_to_cache(entry)
        
        return results
    
    def _numpy_search(
        self,
        query: np.ndarray,
        k: int,
        filter_fn: callable,
    ) -> List[Tuple[MemoryEntry, float]]:
        """Simple numpy-based search"""
        if len(self.entries) == 0:
            return []
        
        vectors = np.stack(self.vectors)
        query = query / np.linalg.norm(query)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        similarities = np.dot(vectors, query)
        
        results = []
        for idx in np.argsort(similarities)[::-1]:
            entry = self.entries[idx]
            if filter_fn(entry):
                results.append((entry, similarities[idx]))
                if len(results) >= k:
                    break
        
        return results
    
    def _add_to_cache(self, entry: MemoryEntry):
        """Add entry to LRU cache"""
        if entry.id in self.cache:
            self.cache_order.remove(entry.id)
        elif len(self.cache_order) >= self.cache_order.maxlen:
            oldest = self.cache_order.popleft()
            del self.cache[oldest]
        
        self.cache[entry.id] = entry
        self.cache_order.append(entry.id)
    
    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get entry by ID (checks cache first)"""
        # Check cache
        if entry_id in self.cache:
            entry = self.cache[entry_id]
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            return entry
        
        # Check primary store
        if self.primary_store:
            entry = self.primary_store.get(entry_id)
            if entry:
                self._add_to_cache(entry)
            return entry
        
        return None
    
    def save(self, path: Optional[str] = None):
        """Save memory store"""
        path = path or self.persist_dir
        os.makedirs(path, exist_ok=True)
        
        if self.primary_store and hasattr(self.primary_store, 'save'):
            self.primary_store.save(path)
        
        # Save cache
        cache_data = {
            entry_id: entry.to_dict()
            for entry_id, entry in self.cache.items()
        }
        with open(os.path.join(path, "cache.json"), "w") as f:
            json.dump(cache_data, f, default=str)
    
    def load(self, path: Optional[str] = None):
        """Load memory store"""
        path = path or self.persist_dir
        
        if self.primary_store and hasattr(self.primary_store, 'load'):
            self.primary_store.load(path)
    
    def __len__(self) -> int:
        if self.primary_store:
            return len(self.primary_store)
        return len(self.entries)
