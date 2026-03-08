"""
Spike-Timing-Dependent Plasticity (STDP)
脉冲时间依赖可塑性

Implements various STDP learning rules for online learning in SNNs:
- Classic STDP
- Reward-modulated STDP (R-STDP)
- Triplet STDP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Callable
import math


class STDPLearner(nn.Module):
    """
    Classic STDP learning rule
    经典STDP学习规则
    
    Δw = A_plus * exp(-Δt/tau_plus) if Δt > 0 (pre before post)
    Δw = -A_minus * exp(Δt/tau_minus) if Δt < 0 (post before pre)
    """
    
    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
        weight_dependence: str = "softbound",  # "softbound", "hardbound", "none"
    ):
        super().__init__()
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
        self.w_min = w_min
        self.weight_dependence = weight_dependence
        
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,  # [batch, time, pre_neurons]
        post_spikes: torch.Tensor,  # [batch, time, post_neurons]
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute weight update using STDP
        
        Args:
            weights: Current weights [post_neurons, pre_neurons]
            pre_spikes: Pre-synaptic spike train
            post_spikes: Post-synaptic spike train
            dt: Time step
            
        Returns:
            delta_w: Weight update
        """
        batch_size, time_steps, pre_n = pre_spikes.shape
        _, _, post_n = post_spikes.shape
        
        # Compute spike traces (exponential decay)
        pre_trace = self.compute_trace(pre_spikes, self.tau_plus, dt)  # [batch, time, pre]
        post_trace = self.compute_trace(post_spikes, self.tau_minus, dt)  # [batch, time, post]
        
        # Compute weight updates
        # LTP: pre fires before post
        # LTD: post fires before pre
        
        delta_w = torch.zeros_like(weights)
        
        for t in range(time_steps):
            # LTP contribution
            # When post spikes, look at pre trace
            post_spike_t = post_spikes[:, t, :].unsqueeze(2)  # [batch, post, 1]
            pre_trace_t = pre_trace[:, t, :].unsqueeze(1)  # [batch, 1, pre]
            
            ltp = self.A_plus * post_spike_t * pre_trace_t  # [batch, post, pre]
            
            # LTD contribution
            pre_spike_t = pre_spikes[:, t, :].unsqueeze(1)  # [batch, 1, pre]
            post_trace_t = post_trace[:, t, :].unsqueeze(2)  # [batch, post, 1]
            
            ltd = -self.A_minus * pre_spike_t * post_trace_t  # [batch, post, pre]
            
            # Weight dependence
            if self.weight_dependence == "softbound":
                # Soft bounds: smaller changes near boundaries
                w_factor = (self.w_max - weights) * (weights - self.w_min) / ((self.w_max - self.w_min) / 2) ** 2
                ltp = ltp * w_factor.unsqueeze(0)
                ltd = ltd * w_factor.unsqueeze(0)
            
            delta_w = delta_w + (ltp + ltd).mean(dim=0)  # Average over batch
        
        # Hard bounds
        if self.weight_dependence == "hardbound":
            new_weights = torch.clamp(weights + delta_w, self.w_min, self.w_max)
            delta_w = new_weights - weights
        
        return delta_w
    
    def compute_trace(
        self,
        spikes: torch.Tensor,
        tau: float,
        dt: float,
    ) -> torch.Tensor:
        """Compute exponential trace of spikes"""
        batch_size, time_steps, num_neurons = spikes.shape
        decay = math.exp(-dt / tau)
        
        trace = torch.zeros_like(spikes)
        for t in range(1, time_steps):
            trace[:, t, :] = decay * trace[:, t-1, :] + spikes[:, t, :]
        
        return trace


class TripletSTDPLearner(STDPLearner):
    """
    Triplet STDP learning rule
    三联体STDP学习规则
    
    Extends classic STDP with third-order spike interactions
    for more stable and biologically plausible learning.
    """
    
    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        A_plus_3: float = 0.005,
        A_minus_3: float = 0.005,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_x: float = 100.0,
        tau_y: float = 100.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
    ):
        super().__init__(A_plus, A_minus, tau_plus, tau_minus, w_max, w_min)
        self.A_plus_3 = A_plus_3
        self.A_minus_3 = A_minus_3
        self.tau_x = tau_x
        self.tau_y = tau_y
    
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Triplet STDP update"""
        batch_size, time_steps, pre_n = pre_spikes.shape
        _, _, post_n = post_spikes.shape
        
        # Compute traces
        pre_trace = self.compute_trace(pre_spikes, self.tau_plus, dt)
        post_trace = self.compute_trace(post_spikes, self.tau_minus, dt)
        pre_trace_slow = self.compute_trace(pre_spikes, self.tau_x, dt)
        post_trace_slow = self.compute_trace(post_spikes, self.tau_y, dt)
        
        delta_w = torch.zeros_like(weights)
        
        for t in range(time_steps):
            # LTP with triplet term
            post_spike_t = post_spikes[:, t, :].unsqueeze(2)
            pre_trace_t = pre_trace[:, t, :].unsqueeze(1)
            post_trace_slow_t = post_trace_slow[:, t, :].unsqueeze(2)
            
            ltp = (self.A_plus + self.A_plus_3 * post_trace_slow_t) * post_spike_t * pre_trace_t
            
            # LTD with triplet term
            pre_spike_t = pre_spikes[:, t, :].unsqueeze(1)
            post_trace_t = post_trace[:, t, :].unsqueeze(2)
            pre_trace_slow_t = pre_trace_slow[:, t, :].unsqueeze(1)
            
            ltd = -(self.A_minus + self.A_minus_3 * pre_trace_slow_t) * pre_spike_t * post_trace_t
            
            delta_w = delta_w + (ltp + ltd).mean(dim=0)
        
        return delta_w


class RewardModulatedSTDPLearner(STDPLearner):
    """
    Reward-modulated STDP (R-STDP)
    奖励调制STDP
    
    STDP gated by reward signal for reinforcement learning.
    """
    
    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_reward: float = 100.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
        baseline: float = 0.0,
    ):
        super().__init__(A_plus, A_minus, tau_plus, tau_minus, w_max, w_min)
        self.tau_reward = tau_reward
        self.baseline = baseline
        
        # Eligibility trace
        self.register_buffer("eligibility", None)
        
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        reward: torch.Tensor,  # [batch] or scalar
        dt: float = 1.0,
    ) -> torch.Tensor:
        """
        R-STDP update
        
        Args:
            weights: Current weights
            pre_spikes: Pre-synaptic spikes
            post_spikes: Post-synaptic spikes
            reward: Reward signal
            dt: Time step
        """
        batch_size, time_steps, pre_n = pre_spikes.shape
        _, _, post_n = post_spikes.shape
        
        # Compute eligibility trace using standard STDP
        eligibility = torch.zeros_like(weights)
        
        pre_trace = self.compute_trace(pre_spikes, self.tau_plus, dt)
        post_trace = self.compute_trace(post_spikes, self.tau_minus, dt)
        
        for t in range(time_steps):
            post_spike_t = post_spikes[:, t, :].unsqueeze(2)
            pre_trace_t = pre_trace[:, t, :].unsqueeze(1)
            ltp = self.A_plus * post_spike_t * pre_trace_t
            
            pre_spike_t = pre_spikes[:, t, :].unsqueeze(1)
            post_trace_t = post_trace[:, t, :].unsqueeze(2)
            ltd = -self.A_minus * pre_spike_t * post_trace_t
            
            eligibility = eligibility + (ltp + ltd).mean(dim=0)
        
        # Modulate by reward
        if isinstance(reward, torch.Tensor):
            if reward.dim() == 0:
                reward = reward.item()
            else:
                reward = reward.mean().item()
        
        modulated_reward = reward - self.baseline
        delta_w = modulated_reward * eligibility
        
        # Update baseline
        self.baseline = 0.99 * self.baseline + 0.01 * reward
        
        return delta_w


class OnlineSTDPLearner(nn.Module):
    """
    Online STDP learner for real-time learning
    在线STDP学习器 - 支持实时流式学习
    
    Features:
    - Streaming weight updates
    - Efficient trace computation
    - Compatible with high-refresh-rate processing
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        A_plus: float = 0.01,
        A_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        tau_trace: float = 100.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
        dt: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.tau_trace = tau_trace
        self.w_max = w_max
        self.w_min = w_min
        self.dt = dt
        
        # Decay constants
        self.decay_pre = math.exp(-dt / tau_plus)
        self.decay_post = math.exp(-dt / tau_minus)
        self.decay_trace = math.exp(-dt / tau_trace)
        
        # Persistent traces for online learning
        self.register_buffer("pre_trace", torch.zeros(in_features))
        self.register_buffer("post_trace", torch.zeros(out_features))
        self.register_buffer("eligibility_trace", torch.zeros(out_features, in_features))
        
    def forward(
        self,
        weights: torch.Tensor,
        pre_spikes: torch.Tensor,  # [batch, in_features]
        post_spikes: torch.Tensor,  # [batch, out_features]
        reward: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Online STDP update
        
        Args:
            weights: Current weights [out, in]
            pre_spikes: Current pre-synaptic spikes
            post_spikes: Current post-synaptic spikes
            reward: Optional reward signal
            
        Returns:
            delta_w: Weight update
        """
        batch_size = pre_spikes.shape[0]
        
        # Update traces (exponential decay + new spikes)
        self.pre_trace = self.decay_pre * self.pre_trace + pre_spikes.mean(dim=0)
        self.post_trace = self.decay_post * self.post_trace + post_spikes.mean(dim=0)
        
        # Compute STDP update
        # LTP: post spikes * pre trace
        ltp = self.A_plus * torch.outer(self.post_trace, self.pre_trace)
        
        # LTD: pre spikes * post trace
        ltd = -self.A_minus * torch.outer(pre_spikes.mean(dim=0), self.post_trace)
        
        # Instantaneous STDP
        instant_stdp = torch.outer(post_spikes.mean(dim=0), self.pre_trace) * self.A_plus
        instant_stdp = instant_stdp - torch.outer(self.post_trace, pre_spikes.mean(dim=0)) * self.A_minus
        
        # Update eligibility trace
        self.eligibility_trace = self.decay_trace * self.eligibility_trace + instant_stdp
        
        # Compute weight update
        if reward is not None:
            # Reward-modulated
            delta_w = reward * self.eligibility_trace
        else:
            # Pure STDP
            delta_w = ltp + ltd
        
        # Apply bounds
        new_weights = torch.clamp(weights + delta_w, self.w_min, self.w_max)
        delta_w = new_weights - weights
        
        return delta_w
    
    def reset_traces(self):
        """Reset all traces"""
        self.pre_trace.zero_()
        self.post_trace.zero_()
        self.eligibility_trace.zero_()


class STDPScheduler:
    """
    Scheduler for STDP learning rate
    STDP学习率调度器
    """
    
    def __init__(
        self,
        learner: STDPLearner,
        mode: str = "cosine",  # "cosine", "exponential", "step"
        T_max: int = 1000,
        eta_min: float = 0.0,
        gamma: float = 0.95,
    ):
        self.learner = learner
        self.mode = mode
        self.T_max = T_max
        self.eta_min = eta_min
        self.gamma = gamma
        self.t = 0
        
        # Store initial values
        self.A_plus_init = learner.A_plus
        self.A_minus_init = learner.A_minus
    
    def step(self):
        """Update learning rates"""
        self.t += 1
        
        if self.mode == "cosine":
            progress = self.t / self.T_max
            factor = self.eta_min + 0.5 * (1 - self.eta_min) * (1 + math.cos(math.pi * progress))
        elif self.mode == "exponential":
            factor = self.gamma ** self.t
        elif self.mode == "step":
            factor = self.gamma ** (self.t // 100)
        else:
            factor = 1.0
        
        self.learner.A_plus = self.A_plus_init * factor
        self.learner.A_minus = self.A_minus_init * factor
    
    def reset(self):
        """Reset scheduler"""
        self.t = 0
        self.learner.A_plus = self.A_plus_init
        self.learner.A_minus = self.A_minus_init
