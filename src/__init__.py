"""
Brain-Inspired AI Package
类脑人工智能包

A high-refresh-rate brain-inspired computing system with:
- Spiking Neural Networks (SNN)
- Online STDP Learning
- Hippocampal Memory System
- Multimodal Processing
- Tool Integration
"""

__version__ = "1.0.0"
__author__ = "Brain-Inspired AI Team"

from .core.neurons import LIFNeuron, AdaptiveLIFNeuron, SpikingLayer
from .core.stdp import STDPLearner, OnlineSTDPLearner, RewardModulatedSTDPLearner
from .memory.hippocampus import HippocampalMemory, MemoryEngram
from .memory.vector_store import UnifiedMemoryStore, MemoryEntry
from .models.model_integration import BrainInspiredModel, StreamToken
from .inference.streaming_inference import StreamingInference, InferencePipeline

__all__ = [
    "LIFNeuron",
    "AdaptiveLIFNeuron",
    "SpikingLayer",
    "STDPLearner",
    "OnlineSTDPLearner",
    "RewardModulatedSTDPLearner",
    "HippocampalMemory",
    "MemoryEngram",
    "UnifiedMemoryStore",
    "MemoryEntry",
    "BrainInspiredModel",
    "StreamToken",
    "StreamingInference",
    "InferencePipeline",
]
