"""
Hippocampus Memory Module
海马体记忆模块

Implements hippocampus-inspired memory system with:
- Pattern separation (dentate gyrus)
- Pattern completion (CA3)
- Consolidation to neocortex
- Neurogenesis
- Episodic and semantic memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from collections import deque
import heapq


@dataclass
class MemoryEngram:
    """
    Memory engram (memory trace)
    记忆印迹
    """
    id: str
    content: torch.Tensor  # Embedding vector
    raw_content: Any  # Original content
    timestamp: datetime
    memory_type: str  # "episodic", "semantic", "procedural"
    
    # Hippocampal features
    context: Dict[str, Any] = field(default_factory=dict)
    emotional_tag: float = 0.0  # -1 to 1
    replay_count: int = 0
    consolidation_level: float = 0.0  # 0 to 1
    
    # Associations
    associated_ids: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)


class DentateGyrus(nn.Module):
    """
    Dentate Gyrus - Pattern Separation
    齿状回 - 模式分离
    
    Expands and orthogonalizes input patterns for distinct memory encoding.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sparsity: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sparsity = sparsity
        
        # Expansion projection
        self.expansion = nn.Linear(input_dim, output_dim * 4, bias=False)
        nn.init.orthogonal_(self.expansion.weight)
        
        # Inhibition for sparse coding
        self.inhibition = nn.Linear(output_dim * 4, output_dim * 4, bias=False)
        nn.init.eye_(self.inhibition.weight)
        
        # Output projection
        self.output = nn.Linear(output_dim * 4, output_dim, bias=False)
        nn.init.orthogonal_(self.output.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pattern separation"""
        # Expand
        h = self.expansion(x)
        h = F.relu(h)
        
        # Competitive inhibition for sparsity
        h = h - self.inhibition(h)
        h = F.relu(h)
        
        # Top-k sparsification
        if self.training:
            k = int(self.sparsity * h.shape[-1])
            topk_vals, topk_idx = torch.topk(h, k, dim=-1)
            h_sparse = torch.zeros_like(h)
            h_sparse.scatter_(-1, topk_idx, topk_vals)
            h = h_sparse
        
        # Project to output
        out = self.output(h)
        return F.normalize(out, p=2, dim=-1)


class CA3Region(nn.Module):
    """
    CA3 Region - Pattern Completion and Sequence Learning
    CA3区域 - 模式补全和序列学习
    
    Auto-associative network for pattern completion and sequence prediction.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        recurrent_steps: int = 3,
    ):
        super().__init__()
        self.dim = dim
        self.recurrent_steps = recurrent_steps
        
        # Auto-associative attention
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Recurrent connections
        self.recurrent = nn.GRUCell(dim, dim)
        
        # Place cell encoding
        self.place_encoding = nn.Linear(dim, dim)
        
        # Sequence prediction
        self.sequence_pred = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pattern completion
        
        Args:
            x: Partial or noisy input
            context: Additional context
            steps: Number of recurrent steps
            
        Returns:
            completed: Completed pattern
            predicted: Next pattern in sequence
        """
        steps = steps or self.recurrent_steps
        
        # Add place encoding
        h = self.place_encoding(x)
        
        # Recurrent pattern completion
        hidden = h
        for _ in range(steps):
            # Self-attention for auto-association
            if context is not None:
                attn_out, _ = self.self_attn(hidden, context, context)
            else:
                attn_out, _ = self.self_attn(hidden, hidden, hidden)
            
            # Recurrent update
            hidden = self.recurrent(attn_out.squeeze(1), hidden.squeeze(1)).unsqueeze(1)
        
        completed = F.normalize(hidden, p=2, dim=-1)
        
        # Sequence prediction
        predicted = self.sequence_pred(completed)
        predicted = F.normalize(predicted, p=2, dim=-1)
        
        return completed, predicted


class CA1Region(nn.Module):
    """
    CA1 Region - Output and Consolidation
    CA1区域 - 输出和巩固
    
    Produces output for neocortex and handles memory consolidation.
    """
    
    def __init__(
        self,
        ca3_dim: int,
        output_dim: int,
    ):
        super().__init__()
        
        self.ca3_to_output = nn.Sequential(
            nn.Linear(ca3_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Linear(output_dim * 2, output_dim),
        )
        
        # Consolidation gate
        self.consolidation_gate = nn.Linear(ca3_dim, 1)
        
    def forward(self, ca3_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process CA3 output
        
        Returns:
            output: Neocortical output
            consolidation_signal: Signal for memory consolidation
        """
        output = self.ca3_to_output(ca3_output)
        output = F.normalize(output, p=2, dim=-1)
        
        consolidation_signal = torch.sigmoid(self.consolidation_gate(ca3_output))
        
        return output, consolidation_signal


class HippocampalMemory(nn.Module):
    """
    Complete Hippocampal Memory System
    完整的海马体记忆系统
    
    Integrates DG, CA3, CA1 for comprehensive memory function.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        dg_dim: int = 2048,
        ca3_dim: int = 1024,
        output_dim: int = 768,
        max_engrams: int = 100000,
        neurogenesis_rate: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_engrams = max_engrams
        self.neurogenesis_rate = neurogenesis_rate
        
        # Hippocampal regions
        self.dentate_gyrus = DentateGyrus(input_dim, dg_dim)
        self.ca3 = CA3Region(ca3_dim)
        self.ca1 = CA1Region(ca3_dim, output_dim)
        
        # Projection between regions
        self.dg_to_ca3 = nn.Linear(dg_dim, ca3_dim)
        self.ca3_to_dg = nn.Linear(ca3_dim, dg_dim)
        
        # Memory storage
        self.engrams: Dict[str, MemoryEngram] = {}
        self.engram_index: List[str] = []
        
        # Embedding matrix for fast similarity search
        self.register_buffer("engram_embeddings", torch.zeros(0, output_dim))
        
        # Neurogenesis tracking
        self.engram_counter = 0
        
    def encode(
        self,
        content: torch.Tensor,
        raw_content: Any = None,
        memory_type: str = "episodic",
        context: Dict[str, Any] = None,
        emotional_tag: float = 0.0,
    ) -> MemoryEngram:
        """
        Encode new memory
        
        Args:
            content: Input embedding
            raw_content: Original content
            memory_type: Type of memory
            context: Contextual information
            emotional_tag: Emotional valence
            
        Returns:
            engram: Created memory engram
        """
        # Pattern separation in DG
        dg_pattern = self.dentate_gyrus(content)
        
        # Project to CA3
        ca3_input = self.dg_to_ca3(dg_pattern)
        
        # Pattern completion in CA3
        ca3_pattern, _ = self.ca3(ca3_input)
        
        # Output via CA1
        output, consolidation = self.ca1(ca3_pattern)
        
        # Create engram
        self.engram_counter += 1
        engram_id = f"engram_{self.engram_counter}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        engram = MemoryEngram(
            id=engram_id,
            content=output.squeeze(0).detach(),
            raw_content=raw_content,
            timestamp=datetime.now(),
            memory_type=memory_type,
            context=context or {},
            emotional_tag=emotional_tag,
            consolidation_level=consolidation.item(),
        )
        
        # Store engram
        self.engrams[engram_id] = engram
        self.engram_index.append(engram_id)
        
        # Update embedding matrix
        new_embedding = output.squeeze(0).detach().unsqueeze(0)
        if self.engram_embeddings.shape[0] == 0:
            self.engram_embeddings = new_embedding
        else:
            self.engram_embeddings = torch.cat([self.engram_embeddings, new_embedding], dim=0)
        
        # Neurogenesis: remove old memories if capacity exceeded
        if len(self.engrams) > self.max_engrams:
            self._forget_oldest()
        
        return engram
    
    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        temporal_weight: float = 0.3,
        semantic_weight: float = 0.7,
    ) -> List[Tuple[MemoryEngram, float]]:
        """
        Retrieve memories by similarity
        
        Args:
            query: Query embedding
            top_k: Number of results
            similarity_threshold: Minimum similarity
            temporal_weight: Weight for temporal similarity
            semantic_weight: Weight for semantic similarity
            
        Returns:
            List of (engram, score) tuples
        """
        if len(self.engrams) == 0:
            return []
        
        # Process query through hippocampus
        dg_pattern = self.dentate_gyrus(query)
        ca3_input = self.dg_to_ca3(dg_pattern)
        ca3_pattern, _ = self.ca3(ca3_input)
        query_embedding, _ = self.ca1(ca3_pattern)
        query_embedding = query_embedding.squeeze(0)
        
        # Semantic similarity
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(0),
            self.engram_embeddings,
            dim=-1
        )
        
        # Temporal similarity (recency bias)
        now = datetime.now()
        temporal_scores = torch.tensor([
            1.0 / (1.0 + (now - self.engrams[eid].timestamp).total_seconds() / 3600)
            for eid in self.engram_index
        ], device=query.device)
        
        # Combined score
        combined_scores = semantic_weight * similarities + temporal_weight * temporal_scores
        
        # Filter by threshold
        valid_mask = similarities >= similarity_threshold
        combined_scores = combined_scores * valid_mask.float()
        
        # Get top-k
        topk_values, topk_indices = torch.topk(combined_scores, min(top_k, len(combined_scores)))
        
        results = []
        for idx, score in zip(topk_indices.tolist(), topk_values.tolist()):
            if score > 0:
                engram_id = self.engram_index[idx]
                engram = self.engrams[engram_id]
                engram.replay_count += 1  # Update replay count
                results.append((engram, score))
        
        return results
    
    def pattern_completion(
        self,
        partial_pattern: torch.Tensor,
        num_iterations: int = 5,
    ) -> torch.Tensor:
        """
        Complete partial memory pattern
        
        Args:
            partial_pattern: Incomplete pattern
            num_iterations: Number of completion iterations
            
        Returns:
            completed: Completed pattern
        """
        # Process through hippocampus
        dg_pattern = self.dentate_gyrus(partial_pattern)
        ca3_input = self.dg_to_ca3(dg_pattern)
        
        # Pattern completion in CA3
        completed, _ = self.ca3(ca3_input, steps=num_iterations)
        
        # Output
        output, _ = self.ca1(completed)
        
        return output
    
    def consolidate(self, engram_ids: Optional[List[str]] = None):
        """
        Consolidate memories to long-term storage
        
        Args:
            engram_ids: Specific engrams to consolidate, or None for all
        """
        ids_to_consolidate = engram_ids or list(self.engrams.keys())
        
        for eid in ids_to_consolidate:
            if eid in self.engrams:
                engram = self.engrams[eid]
                # Increase consolidation level
                engram.consolidation_level = min(1.0, engram.consolidation_level + 0.1)
    
    def associate(self, engram_id1: str, engram_id2: str):
        """Create association between two memories"""
        if engram_id1 in self.engrams and engram_id2 in self.engrams:
            if engram_id2 not in self.engrams[engram_id1].associated_ids:
                self.engrams[engram_id1].associated_ids.append(engram_id2)
            if engram_id1 not in self.engrams[engram_id2].associated_ids:
                self.engrams[engram_id2].associated_ids.append(engram_id1)
    
    def _forget_oldest(self, num_to_forget: int = 1):
        """Forget oldest, least consolidated memories"""
        # Score memories for forgetting
        now = datetime.now()
        forget_scores = []
        
        for eid, engram in self.engrams.items():
            age = (now - engram.timestamp).total_seconds()
            consolidation = engram.consolidation_level
            replay = engram.replay_count
            
            # Higher score = more likely to forget
            score = age / 86400 - consolidation * 10 - replay * 0.1
            forget_scores.append((score, eid))
        
        # Remove lowest priority memories
        forget_scores.sort(reverse=True)
        for _, eid in forget_scores[:num_to_forget]:
            self._remove_engram(eid)
    
    def _remove_engram(self, engram_id: str):
        """Remove an engram from memory"""
        if engram_id in self.engrams:
            idx = self.engram_index.index(engram_id)
            self.engram_index.pop(idx)
            del self.engrams[engram_id]
            
            # Update embedding matrix
            mask = torch.ones(self.engram_embeddings.shape[0], dtype=torch.bool)
            mask[idx] = False
            self.engram_embeddings = self.engram_embeddings[mask]
    
    def save(self, path: str):
        """Save memory system"""
        state = {
            "engrams": self.engrams,
            "engram_index": self.engram_index,
            "engram_embeddings": self.engram_embeddings,
            "engram_counter": self.engram_counter,
            "state_dict": self.state_dict(),
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load memory system"""
        state = torch.load(path, map_location="cpu")
        self.engrams = state["engrams"]
        self.engram_index = state["engram_index"]
        self.engram_embeddings = state["engram_embeddings"]
        self.engram_counter = state["engram_counter"]
        self.load_state_dict(state["state_dict"])


class MemoryIndex:
    """
    Fast memory index for similarity search
    快速记忆索引
    """
    
    def __init__(self, dim: int, index_type: str = "flat"):
        self.dim = dim
        self.index_type = index_type
        self.vectors = []
        self.ids = []
        
    def add(self, vector: np.ndarray, id: str):
        """Add vector to index"""
        self.vectors.append(vector)
        self.ids.append(id)
    
    def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors"""
        if len(self.vectors) == 0:
            return []
        
        vectors = np.stack(self.vectors)
        similarities = np.dot(vectors, query) / (
            np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
        )
        
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        return [(self.ids[i], similarities[i]) for i in top_k_idx]
