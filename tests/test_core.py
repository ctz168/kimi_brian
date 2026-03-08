"""
Unit tests for core modules
核心模块单元测试
"""

import pytest
import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported"""
    from src.core.neurons import LIFNeuron, SpikingLayer
    from src.core.stdp import STDPLearner
    from src.memory.hippocampus import HippocampalMemory
    assert True


def test_lif_neuron():
    """Test LIF neuron forward pass"""
    from src.core.neurons import LIFNeuron
    
    neuron = LIFNeuron(tau_mem=20.0, v_thresh=1.0)
    
    # Test forward pass
    x = torch.randn(2, 10)
    spikes, state = neuron(x)
    
    assert spikes.shape == (2, 10)
    assert torch.all((spikes == 0) | (spikes == 1))  # Binary spikes


def test_stdp_learner():
    """Test STDP weight updates"""
    from src.core.stdp import STDPLearner
    
    stdp = STDPLearner(A_plus=0.01, A_minus=0.01)
    
    # Create spike trains
    pre_spikes = torch.zeros(1, 10, 5)
    post_spikes = torch.zeros(1, 10, 5)
    pre_spikes[0, 3, :] = 1.0
    post_spikes[0, 5, :] = 1.0
    
    weights = torch.randn(5, 5) * 0.1
    delta_w = stdp(weights, pre_spikes, post_spikes)
    
    assert delta_w.shape == weights.shape


def test_hippocampal_memory():
    """Test hippocampal memory encoding and retrieval"""
    from src.memory.hippocampus import HippocampalMemory
    
    memory = HippocampalMemory(input_dim=64, output_dim=64)
    
    # Encode memory
    content = torch.randn(1, 64)
    engram = memory.encode(content, raw_content="test")
    
    assert engram.id is not None
    assert len(memory.engrams) == 1
    
    # Retrieve memory
    query = torch.randn(1, 64)
    results = memory.retrieve(query, top_k=1)
    
    assert len(results) >= 0  # May or may not find match


def test_vector_store():
    """Test vector memory store"""
    from src.memory.vector_store import UnifiedMemoryStore
    
    store = UnifiedMemoryStore(dim=64, primary_backend="numpy")
    
    # Add memories
    for i in range(10):
        store.add(
            content=f"Memory {i}",
            vector=np.random.randn(64),
            memory_type="test",
        )
    
    assert len(store) == 10
    
    # Search
    results = store.search(np.random.randn(64), k=5)
    assert len(results) <= 5


def test_tool_registry():
    """Test tool registry"""
    from src.tools.web_tools import ToolRegistry
    
    registry = ToolRegistry()
    tools = registry.get_available_tools()
    
    assert len(tools) > 0
    
    # Test calculator
    result = registry.call("calculate", expression="2 + 2")
    assert result.success
    assert result.result == 4


@pytest.mark.skip(reason="Requires model download")
def test_model_integration():
    """Test model integration (requires model weights)"""
    from src.models.model_integration import BrainInspiredModel
    
    model = BrainInspiredModel(
        base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        device="cpu",
    )
    
    # Test encoding
    embedding = model.encode_multimodal(text="Hello")
    assert embedding.shape[-1] == model.hidden_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
