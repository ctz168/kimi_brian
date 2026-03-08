"""
Online Learning Module
在线学习模块

Implements real-time learning capabilities:
- Stream-based learning
- STDP weight updates
- Memory consolidation
- Reward-based learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import threading
from queue import Queue
import numpy as np

from ..core.stdp import OnlineSTDPLearner, RewardModulatedSTDPLearner
from ..memory.hippocampus import MemoryEngram


@dataclass
class LearningExperience:
    """Single learning experience"""
    input_data: torch.Tensor
    output_data: Optional[torch.Tensor] = None
    reward: Optional[float] = None
    timestamp: datetime = None
    context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context is None:
            self.context = {}


class OnlineLearner:
    """
    Online learner for real-time learning
    在线学习器
    
    Features:
    - Continuous learning from stream
    - STDP-based weight updates
    - Reward modulation
    - Experience replay
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        buffer_size: int = 1000,
        batch_size: int = 32,
        update_interval: int = 100,
        use_stdp: bool = True,
        use_replay: bool = True,
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.use_stdp = use_stdp
        self.use_replay = use_replay
        
        # Experience buffer
        self.experience_buffer: List[LearningExperience] = []
        
        # STDP learners
        self.stdp_learners: Dict[str, OnlineSTDPLearner] = {}
        if use_stdp:
            self._init_stdp_learners()
        
        # Optimizer for backprop
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
        
        # Learning statistics
        self.stats = {
            "total_experiences": 0,
            "updates": 0,
            "average_reward": 0.0,
        }
        
        # Threading
        self.lock = threading.Lock()
        self.learning_queue = Queue()
        self.is_learning = False
    
    def _init_stdp_learners(self):
        """Initialize STDP learners for spiking layers"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'linear') and 'snn' in name.lower():
                in_features = module.linear.in_features
                out_features = module.linear.out_features
                
                self.stdp_learners[name] = OnlineSTDPLearner(
                    in_features=in_features,
                    out_features=out_features,
                )
    
    def learn(
        self,
        experience: LearningExperience,
        immediate_update: bool = False,
    ) -> Dict[str, float]:
        """
        Learn from a single experience
        
        Args:
            experience: Learning experience
            immediate_update: Whether to update immediately
            
        Returns:
            Learning metrics
        """
        with self.lock:
            # Add to buffer
            self.experience_buffer.append(experience)
            self.stats["total_experiences"] += 1
            
            # Maintain buffer size
            if len(self.experience_buffer) > self.buffer_size:
                self.experience_buffer.pop(0)
            
            # Update reward statistics
            if experience.reward is not None:
                self.stats["average_reward"] = (
                    0.99 * self.stats["average_reward"] +
                    0.01 * experience.reward
                )
            
            metrics = {}
            
            # STDP update
            if self.use_stdp:
                stdp_metrics = self._stdp_update(experience)
                metrics.update(stdp_metrics)
            
            # Backprop update
            if immediate_update or len(self.experience_buffer) >= self.update_interval:
                bp_metrics = self._backprop_update()
                metrics.update(bp_metrics)
            
            return metrics
    
    def _stdp_update(self, experience: LearningExperience) -> Dict[str, float]:
        """STDP-based weight update"""
        metrics = {}
        
        # Convert input to spikes
        input_spikes = self._to_spikes(experience.input_data)
        
        if experience.output_data is not None:
            output_spikes = self._to_spikes(experience.output_data)
        else:
            # Forward pass to get output
            with torch.no_grad():
                output = self.model(experience.input_data)
                output_spikes = self._to_spikes(output)
        
        # Apply STDP to each layer
        for name, learner in self.stdp_learners.items():
            module = dict(self.model.named_modules())[name]
            
            # Get reward signal
            reward = experience.reward if experience.reward is not None else 0.0
            
            # Compute weight update
            delta_w = learner(
                weights=module.linear.weight,
                pre_spikes=input_spikes,
                post_spikes=output_spikes,
                reward=torch.tensor(reward),
            )
            
            # Apply update
            module.linear.weight.data += delta_w * self.learning_rate
            
            metrics[f"{name}_dw_norm"] = delta_w.norm().item()
        
        return metrics
    
    def _backprop_update(self) -> Dict[str, float]:
        """Backpropagation-based update using experience replay"""
        if not self.use_replay or len(self.experience_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        indices = np.random.choice(
            len(self.experience_buffer),
            size=self.batch_size,
            replace=False,
        )
        
        batch = [self.experience_buffer[i] for i in indices]
        
        # Prepare batch data
        inputs = torch.stack([e.input_data for e in batch])
        
        # Compute loss
        self.optimizer.zero_grad()
        
        outputs = self.model(inputs)
        
        # Reconstruction loss (autoencoder-style)
        loss = F.mse_loss(outputs, inputs)
        
        # Add reward-based loss
        rewards = torch.tensor([e.reward or 0.0 for e in batch])
        reward_loss = -rewards.mean()  # Maximize reward
        
        total_loss = loss + 0.1 * reward_loss
        
        # Backprop
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        self.stats["updates"] += 1
        
        return {
            "loss": loss.item(),
            "reward_loss": reward_loss.item(),
            "total_loss": total_loss.item(),
        }
    
    def _to_spikes(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Convert continuous values to spikes"""
        # Rate coding: probability based on value
        probs = torch.sigmoid(x)
        spikes = (torch.rand_like(probs) < probs).float()
        return spikes
    
    def learn_from_text(
        self,
        text: str,
        embedding: torch.Tensor,
        reward: Optional[float] = None,
    ) -> Dict[str, float]:
        """Learn from text experience"""
        experience = LearningExperience(
            input_data=embedding,
            reward=reward,
            context={"text": text},
        )
        return self.learn(experience)
    
    def learn_from_interaction(
        self,
        user_input: str,
        model_output: str,
        user_feedback: float,  # -1 to 1
    ) -> Dict[str, float]:
        """Learn from user feedback"""
        # Encode interaction
        # This would use the model's encoder in practice
        dummy_embedding = torch.randn(1, 768)  # Placeholder
        
        experience = LearningExperience(
            input_data=dummy_embedding,
            reward=user_feedback,
            context={
                "user_input": user_input,
                "model_output": model_output,
            },
        )
        
        return self.learn(experience, immediate_update=True)
    
    def consolidate_memories(self):
        """Consolidate experiences to long-term memory"""
        # Trigger memory consolidation in the model
        if hasattr(self.model, 'hippocampus'):
            self.model.hippocampus.consolidate()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            **self.stats,
            "buffer_size": len(self.experience_buffer),
            "stdp_learners": len(self.stdp_learners),
        }
    
    def save(self, path: str):
        """Save learner state"""
        state = {
            "optimizer": self.optimizer.state_dict(),
            "stats": self.stats,
            "stdp_learners": {
                name: learner.state_dict()
                for name, learner in self.stdp_learners.items()
            },
        }
        torch.save(state, path)
    
    def load(self, path: str):
        """Load learner state"""
        state = torch.load(path, map_location="cpu")
        self.optimizer.load_state_dict(state["optimizer"])
        self.stats = state["stats"]
        
        for name, learner_state in state["stdp_learners"].items():
            if name in self.stdp_learners:
                self.stdp_learners[name].load_state_dict(learner_state)


class StreamingLearner:
    """
    Streaming learner for high-refresh-rate learning
    流式学习器
    
    Processes learning updates in real-time during inference.
    """
    
    def __init__(
        self,
        online_learner: OnlineLearner,
        update_frequency: int = 10,  # Update every N tokens
    ):
        self.online_learner = online_learner
        self.update_frequency = update_frequency
        self.token_count = 0
        self.recent_hidden_states: List[torch.Tensor] = []
        self.max_hidden_states = 100
    
    def on_token_generated(
        self,
        token: str,
        hidden_state: torch.Tensor,
        memory_context: Optional[List[Dict]] = None,
    ):
        """Called for each generated token"""
        self.token_count += 1
        
        # Store hidden state
        self.recent_hidden_states.append(hidden_state.detach())
        if len(self.recent_hidden_states) > self.max_hidden_states:
            self.recent_hidden_states.pop(0)
        
        # Periodic learning update
        if self.token_count % self.update_frequency == 0:
            self._streaming_update(token, hidden_state)
    
    def _streaming_update(
        self,
        token: str,
        hidden_state: torch.Tensor,
    ):
        """Perform streaming update"""
        if len(self.recent_hidden_states) < 2:
            return
        
        # Create experience from sequence
        prev_state = self.recent_hidden_states[-2]
        curr_state = self.recent_hidden_states[-1]
        
        experience = LearningExperience(
            input_data=prev_state,
            output_data=curr_state,
            context={"token": token},
        )
        
        # Learn
        self.online_learner.learn(experience)
    
    def on_sequence_complete(
        self,
        sequence: str,
        reward: Optional[float] = None,
    ):
        """Called when a sequence generation is complete"""
        # Final update with reward
        if len(self.recent_hidden_states) > 0:
            experience = LearningExperience(
                input_data=self.recent_hidden_states[-1],
                reward=reward,
                context={"sequence": sequence},
            )
            self.online_learner.learn(experience, immediate_update=True)
        
        # Clear state
        self.recent_hidden_states.clear()
        self.token_count = 0
