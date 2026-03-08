"""
Model Integration Module
模型整合模块

Integrates multiple models for brain-inspired AI:
- Qwen2.5 base model for language and reasoning
- CLIP/World model for vision understanding
- SNN for spiking neural computation
- Memory-augmented processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Union, Generator
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    CLIPModel, 
    CLIPProcessor,
    pipeline,
)
from PIL import Image
import numpy as np
from dataclasses import dataclass
import threading
from queue import Queue

from ..core.neurons import SpikingLayer
from ..core.stdp import OnlineSTDPLearner
from ..memory.hippocampus import HippocampalMemory
from ..memory.vector_store import UnifiedMemoryStore


@dataclass
class StreamToken:
    """Streaming token output"""
    token: str
    token_id: int
    is_special: bool = False
    logits: Optional[torch.Tensor] = None
    hidden_state: Optional[torch.Tensor] = None
    memory_context: Optional[List[Dict]] = None


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder for vision, audio, and text
    多模态编码器
    """
    
    def __init__(
        self,
        vision_model_name: str = "openai/clip-vit-base-patch32",
        output_dim: int = 768,
        device: str = "auto",
    ):
        super().__init__()
        self.output_dim = output_dim
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load vision model
        try:
            self.vision_model = CLIPModel.from_pretrained(vision_model_name).to(device)
            self.vision_processor = CLIPProcessor.from_pretrained(vision_model_name)
            self.vision_dim = self.vision_model.config.vision_config.hidden_size
        except Exception as e:
            print(f"Warning: Could not load vision model: {e}")
            self.vision_model = None
            self.vision_processor = None
            self.vision_dim = 512
        
        # Projection layers
        self.vision_proj = nn.Linear(self.vision_dim, output_dim)
        
        # Audio encoder (placeholder for future implementation)
        self.audio_proj = nn.Linear(80, output_dim)  # Mel spectrogram input
        
    def encode_image(
        self,
        images: Union[Image.Image, List[Image.Image]],
    ) -> torch.Tensor:
        """Encode images to embeddings"""
        if self.vision_model is None:
            # Return dummy embeddings
            batch_size = 1 if isinstance(images, Image.Image) else len(images)
            return torch.randn(batch_size, self.output_dim, device=self.device)
        
        if isinstance(images, Image.Image):
            images = [images]
        
        inputs = self.vision_processor(images=images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            vision_outputs = self.vision_model.vision_model(**inputs)
            image_embeds = vision_outputs.pooler_output
        
        return self.vision_proj(image_embeds)
    
    def encode_audio(
        self,
        audio: torch.Tensor,  # Mel spectrogram [batch, time, mel]
    ) -> torch.Tensor:
        """Encode audio to embeddings"""
        # Simple mean pooling (can be replaced with proper audio encoder)
        audio_features = audio.mean(dim=1)  # [batch, mel]
        return self.audio_proj(audio_features)


class BrainInspiredModel(nn.Module):
    """
    Main brain-inspired AI model
    主类脑AI模型
    
    Integrates:
    - Base LLM (Qwen2.5) for language processing
    - Spiking neural network layers
    - Hippocampal memory system
    - Multimodal encoders
    - Online learning capabilities
    """
    
    def __init__(
        self,
        base_model_name: str = "Qwen/Qwen3.5-0.8B",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        device: str = "auto",
        use_snn: bool = True,
        use_memory: bool = True,
        memory_dim: int = 768,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
        print(f"Loading base model: {base_model_name} on {device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if load_in_8bit and device == "cuda":
            model_kwargs["load_in_8bit"] = True
        elif load_in_4bit and device == "cuda":
            model_kwargs["load_in_4bit"] = True
            model_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                **model_kwargs
            )
            if not (load_in_8bit or load_in_4bit):
                self.base_model = self.base_model.to(device)
            self.base_model.eval()
        except Exception as e:
            print(f"Error loading base model: {e}")
            # Fallback to smaller model
            self.base_model = None
        
        self.hidden_size = self.base_model.config.hidden_size if self.base_model else 768
        
        # Multimodal encoder
        self.multimodal_encoder = MultimodalEncoder(
            vision_model_name=vision_model_name,
            output_dim=self.hidden_size,
            device=device,
        )
        
        # Spiking neural network layers
        self.use_snn = use_snn
        if use_snn:
            self.snn_input = SpikingLayer(self.hidden_size, self.hidden_size, neuron_type="LIF")
            self.snn_hidden = SpikingLayer(self.hidden_size, self.hidden_size, neuron_type="AdaptiveLIF")
            self.snn_output = SpikingLayer(self.hidden_size, self.hidden_size, neuron_type="LIF")
            
            # Online STDP learner
            self.stdp_learner = OnlineSTDPLearner(
                self.hidden_size,
                self.hidden_size,
                A_plus=0.01,
                A_minus=0.01,
            )
        
        # Memory systems
        self.use_memory = use_memory
        if use_memory:
            # Hippocampal memory (working/episodic)
            self.hippocampus = HippocampalMemory(
                input_dim=self.hidden_size,
                output_dim=memory_dim,
            )
            
            # Long-term memory store
            self.long_term_memory = UnifiedMemoryStore(
                dim=memory_dim,
                primary_backend="faiss",
            )
        
        # Fusion layer for integrating different modalities
        self.fusion_layer = nn.MultiheadAttention(
            self.hidden_size,
            num_heads=8,
            batch_first=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Streaming state
        self.streaming_state: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
    def encode_multimodal(
        self,
        text: Optional[str] = None,
        images: Optional[List[Image.Image]] = None,
        audio: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode multimodal inputs"""
        embeddings = []
        
        # Encode text
        if text:
            text_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)
            
            with torch.no_grad():
                text_embeds = self.base_model.get_input_embeddings()(text_inputs.input_ids)
                text_embeds = text_embeds.mean(dim=1)  # Pool
            embeddings.append(text_embeds)
        
        # Encode images
        if images:
            image_embeds = self.multimodal_encoder.encode_image(images)
            embeddings.append(image_embeds)
        
        # Encode audio
        if audio is not None:
            audio_embeds = self.multimodal_encoder.encode_audio(audio)
            embeddings.append(audio_embeds)
        
        # Fuse embeddings
        if len(embeddings) == 0:
            return torch.zeros(1, self.hidden_size, device=self.device)
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        # Multi-modal fusion
        stacked = torch.stack([e.mean(dim=0) for e in embeddings], dim=0).unsqueeze(0)
        fused, _ = self.fusion_layer(stacked, stacked, stacked)
        return fused.mean(dim=1)
    
    def retrieve_memory(
        self,
        query: torch.Tensor,
        top_k: int = 5,
    ) -> List[Dict]:
        """Retrieve relevant memories"""
        if not self.use_memory:
            return []
        
        # Query hippocampus
        hippocampal_memories = self.hippocampus.retrieve(
            query,
            top_k=top_k // 2,
        )
        
        # Query long-term memory
        ltm_memories = self.long_term_memory.search(
            query.squeeze(0).cpu().numpy(),
            k=top_k // 2,
        )
        
        # Combine and format
        memories = []
        for engram, score in hippocampal_memories:
            memories.append({
                "content": engram.raw_content,
                "type": "hippocampal",
                "score": score,
                "timestamp": engram.timestamp,
            })
        
        for entry, score in ltm_memories:
            memories.append({
                "content": entry.content,
                "type": "long_term",
                "score": score,
                "timestamp": entry.timestamp,
            })
        
        # Sort by score
        memories.sort(key=lambda x: x["score"], reverse=True)
        
        return memories[:top_k]
    
    def store_memory(
        self,
        content: Any,
        embedding: torch.Tensor,
        memory_type: str = "episodic",
    ):
        """Store in memory"""
        if not self.use_memory:
            return
        
        # Store in hippocampus
        self.hippocampus.encode(
            content=embedding,
            raw_content=content,
            memory_type=memory_type,
        )
        
        # Store in long-term memory if consolidated
        if isinstance(content, str):
            self.long_term_memory.add(
                content=content,
                vector=embedding.squeeze(0).cpu().numpy(),
                memory_type=memory_type,
            )
    
    def process_snn(
        self,
        x: torch.Tensor,
        state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Process through spiking neural network"""
        if not self.use_snn:
            return x, state or {}
        
        state = state or {}
        
        # Convert to spikes (rate coding)
        spike_probs = torch.sigmoid(x)
        spikes = (torch.rand_like(spike_probs) < spike_probs).float()
        
        # SNN layers
        spikes, state["snn_input"] = self.snn_input(spikes, state.get("snn_input"))
        spikes, state["snn_hidden"] = self.snn_hidden(spikes, state.get("snn_hidden"))
        spikes, state["snn_output"] = self.snn_output(spikes, state.get("snn_output"))
        
        # Convert back to continuous
        output = spikes * 2 - 1  # Scale to [-1, 1]
        
        # Online STDP learning
        if self.training:
            with torch.no_grad():
                delta_w = self.stdp_learner(
                    self.snn_input.linear.weight,
                    spikes,
                    spikes,
                )
                self.snn_input.linear.weight.data += delta_w
        
        return output, state
    
    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        use_memory: bool = True,
        use_snn: bool = True,
        images: Optional[List[Image.Image]] = None,
    ) -> Generator[StreamToken, None, None]:
        """
        Stream generation with brain-inspired features
        
        Yields tokens as they are generated for high-refresh-rate output.
        """
        if self.base_model is None:
            yield StreamToken(
                token="Model not loaded properly.",
                token_id=0,
                is_special=True,
            )
            return
        
        # Encode input
        input_embedding = self.encode_multimodal(text=prompt, images=images)
        
        # Retrieve relevant memories
        memories = []
        if use_memory and self.use_memory:
            memories = self.retrieve_memory(input_embedding, top_k=5)
            memory_context = "\n".join([m["content"] for m in memories if m["content"]])
            if memory_context:
                prompt = f"[Relevant context: {memory_context}]\n\n{prompt}"
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids
        
        # Generation state
        past_key_values = None
        snn_state = {}
        
        for i in range(max_new_tokens):
            # Forward pass
            outputs = self.base_model(
                input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            hidden_state = outputs.hidden_states[-1][:, -1, :] if outputs.hidden_states else None
            
            # Apply SNN processing
            if use_snn and self.use_snn and hidden_state is not None:
                hidden_state, snn_state = self.process_snn(hidden_state, snn_state)
                logits = logits + self.output_proj(hidden_state)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-p sampling
            probs = F.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumsum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[0, indices_to_remove] = float('-inf')
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
            
            # Decode token
            token_text = self.tokenizer.decode(next_token_id[0], skip_special_tokens=True)
            
            # Store memory
            if use_memory and hidden_state is not None:
                self.store_memory(
                    content=token_text,
                    embedding=hidden_state,
                    memory_type="episodic",
                )
            
            yield StreamToken(
                token=token_text,
                token_id=next_token_id.item(),
                is_special=next_token_id.item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id],
                logits=logits[0],
                hidden_state=hidden_state[0] if hidden_state is not None else None,
                memory_context=memories if i == 0 else None,
            )
            
            # Check for end of sequence
            if next_token_id.item() == self.tokenizer.eos_token_id:
                break
            
            # Update input for next iteration
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Non-streaming generation"""
        tokens = []
        for token in self.generate_stream(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        ):
            tokens.append(token.token)
        
        return "".join(tokens)
    
    def save(self, path: str):
        """Save model"""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save model components
        state_dict = {
            "multimodal_encoder": self.multimodal_encoder.state_dict(),
            "fusion_layer": self.fusion_layer.state_dict(),
            "output_proj": self.output_proj.state_dict(),
        }
        
        if self.use_snn:
            state_dict["snn_input"] = self.snn_input.state_dict()
            state_dict["snn_hidden"] = self.snn_hidden.state_dict()
            state_dict["snn_output"] = self.snn_output.state_dict()
            state_dict["stdp_learner"] = self.stdp_learner.state_dict()
        
        if self.use_memory:
            state_dict["hippocampus"] = self.hippocampus.state_dict()
        
        torch.save(state_dict, os.path.join(path, "brain_model.pt"))
        
        # Save memories
        if self.use_memory:
            self.hippocampus.save(os.path.join(path, "hippocampus.pt"))
            self.long_term_memory.save(os.path.join(path, "ltm"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
    
    def load(self, path: str):
        """Load model"""
        import os
        
        state_dict = torch.load(
            os.path.join(path, "brain_model.pt"),
            map_location=self.device,
        )
        
        self.multimodal_encoder.load_state_dict(state_dict["multimodal_encoder"])
        self.fusion_layer.load_state_dict(state_dict["fusion_layer"])
        self.output_proj.load_state_dict(state_dict["output_proj"])
        
        if self.use_snn and "snn_input" in state_dict:
            self.snn_input.load_state_dict(state_dict["snn_input"])
            self.snn_hidden.load_state_dict(state_dict["snn_hidden"])
            self.snn_output.load_state_dict(state_dict["snn_output"])
            self.stdp_learner.load_state_dict(state_dict["stdp_learner"])
        
        if self.use_memory and "hippocampus" in state_dict:
            self.hippocampus.load_state_dict(state_dict["hippocampus"])
            self.hippocampus.load(os.path.join(path, "hippocampus.pt"))
            self.long_term_memory.load(os.path.join(path, "ltm"))
