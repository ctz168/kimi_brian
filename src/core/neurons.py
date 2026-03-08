"""
Spiking Neural Network Neurons
脉冲神经网络神经元实现

Implements various neuron models for brain-inspired computing:
- LIF (Leaky Integrate-and-Fire)
- AdEx (Adaptive Exponential)
- Izhikevich
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron with STDP support
    带STDP支持的泄漏整合发放神经元
    
    Features:
    - Membrane potential dynamics
    - Refractory period
    - Spike generation
    - Surrogate gradient for backpropagation
    """
    
    def __init__(
        self,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        v_rest: float = 0.0,
        dt: float = 1.0,
        spike_surrogate: str = "fast_sigmoid",
        beta: float = 10.0,
    ):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v_rest = v_rest
        self.dt = dt
        self.beta = beta
        
        # Time constants in discrete time
        self.alpha = math.exp(-dt / tau_mem)
        self.syn_decay = math.exp(-dt / tau_syn)
        
        # Surrogate gradient function
        self.spike_surrogate = spike_surrogate
        
    def forward(
        self,
        x: torch.Tensor,  # Input current [batch, neurons]
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass of LIF neuron
        
        Args:
            x: Input current
            state: Previous state (v_mem, i_syn, refractory)
            
        Returns:
            spikes: Output spikes
            new_state: Updated state
        """
        batch_size, num_neurons = x.shape
        
        # Initialize state if None
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=x.device, dtype=x.dtype)
            i_syn = torch.zeros(batch_size, num_neurons, device=x.device, dtype=x.dtype)
            refractory = torch.zeros(batch_size, num_neurons, device=x.device, dtype=x.dtype)
        else:
            v_mem = state["v_mem"]
            i_syn = state["i_syn"]
            refractory = state["refractory"]
        
        # Synaptic current dynamics
        i_syn = self.syn_decay * i_syn + x
        
        # Check if neuron is in refractory period
        not_refractory = (refractory <= 0).float()
        
        # Membrane potential dynamics (only update if not refractory)
        dv = (self.v_rest - v_mem) / self.tau_mem + i_syn
        v_mem = v_mem + self.dt * dv * not_refractory
        
        # Spike generation with surrogate gradient
        spikes = self.spike_function(v_mem - self.v_thresh)
        
        # Reset membrane potential after spike
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        
        # Update refractory period
        refractory = torch.where(
            spikes > 0.5,
            torch.ones_like(refractory) * 5,  # 5ms refractory period
            torch.clamp(refractory - 1, min=0)
        )
        
        new_state = {
            "v_mem": v_mem,
            "i_syn": i_syn,
            "refractory": refractory,
        }
        
        return spikes, new_state
    
    def spike_function(self, v_diff: torch.Tensor) -> torch.Tensor:
        """Spike generation with surrogate gradient"""
        if self.spike_surrogate == "fast_sigmoid":
            # Fast sigmoid surrogate
            return FastSigmoid.apply(v_diff, self.beta)
        elif self.spike_surrogate == "arctan":
            return Arctan.apply(v_diff, self.beta)
        elif self.spike_surrogate == "triangle":
            return Triangle.apply(v_diff, self.beta)
        else:
            # Straight-through estimator
            return StraightThrough.apply(v_diff)


class AdaptiveLIFNeuron(nn.Module):
    """
    Adaptive LIF neuron with threshold adaptation
    带阈值自适应的自适应LIF神经元
    """
    
    def __init__(
        self,
        tau_mem: float = 20.0,
        tau_adapt: float = 100.0,
        v_thresh: float = 1.0,
        v_reset: float = 0.0,
        adapt_scale: float = 0.1,
        dt: float = 1.0,
    ):
        super().__init__()
        self.tau_mem = tau_mem
        self.tau_adapt = tau_adapt
        self.v_thresh_base = v_thresh
        self.v_reset = v_reset
        self.adapt_scale = adapt_scale
        self.dt = dt
        
        self.alpha = math.exp(-dt / tau_mem)
        self.adapt_decay = math.exp(-dt / tau_adapt)
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_neurons = x.shape
        
        if state is None:
            v_mem = torch.zeros(batch_size, num_neurons, device=x.device, dtype=x.dtype)
            v_thresh_adapt = torch.zeros(batch_size, num_neurons, device=x.device, dtype=x.dtype)
        else:
            v_mem = state["v_mem"]
            v_thresh_adapt = state["v_thresh_adapt"]
        
        # Effective threshold with adaptation
        v_thresh_eff = self.v_thresh_base + v_thresh_adapt
        
        # Membrane dynamics
        dv = -v_mem / self.tau_mem + x
        v_mem = v_mem + self.dt * dv
        
        # Spike generation
        spikes = (v_mem >= v_thresh_eff).float()
        
        # Reset and adaptation
        v_mem = v_mem * (1 - spikes) + self.v_reset * spikes
        v_thresh_adapt = self.adapt_decay * v_thresh_adapt + self.adapt_scale * spikes
        
        new_state = {
            "v_mem": v_mem,
            "v_thresh_adapt": v_thresh_adapt,
        }
        
        return spikes, new_state


class IzhikevichNeuron(nn.Module):
    """
    Izhikevich neuron model
    Izhikevich神经元模型 - 可以模拟多种神经元类型
    """
    
    def __init__(
        self,
        a: float = 0.02,
        b: float = 0.2,
        c: float = -65.0,
        d: float = 8.0,
        dt: float = 0.5,
    ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.dt = dt
        
    def forward(
        self,
        x: torch.Tensor,
        state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, num_neurons = x.shape
        
        if state is None:
            v = torch.full((batch_size, num_neurons), -65.0, device=x.device, dtype=x.dtype)
            u = torch.full((batch_size, num_neurons), -13.0, device=x.device, dtype=x.dtype)
        else:
            v = state["v"]
            u = state["u"]
        
        # Izhikevich equations
        dv = 0.04 * v**2 + 5 * v + 140 - u + x
        du = self.a * (self.b * v - u)
        
        v = v + self.dt * dv
        u = u + self.dt * du
        
        # Spike detection
        spikes = (v >= 30.0).float()
        
        # Reset
        v = torch.where(v >= 30.0, torch.full_like(v, self.c), v)
        u = torch.where(v >= 30.0, u + self.d, u)
        
        new_state = {"v": v, "u": u}
        
        return spikes, new_state


# Surrogate gradient functions
class FastSigmoid(torch.autograd.Function):
    """Fast sigmoid surrogate gradient"""
    
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output * beta / (1 + beta * input.abs()).pow(2)
        return grad_input, None


class Arctan(torch.autograd.Function):
    """Arctan surrogate gradient"""
    
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output * beta / (1 + (beta * input).pow(2))
        return grad_input, None


class Triangle(torch.autograd.Function):
    """Triangle surrogate gradient"""
    
    @staticmethod
    def forward(ctx, input, beta):
        ctx.save_for_backward(input)
        ctx.beta = beta
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        beta = ctx.beta
        grad_input = grad_output * beta * (1 - beta * input.abs()).clamp(min=0)
        return grad_input, None


class StraightThrough(torch.autograd.Function):
    """Straight-through estimator"""
    
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class SpikingLayer(nn.Module):
    """
    Spiking neural network layer with multiple neurons
    脉冲神经网络层
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        neuron_type: str = "LIF",
        neuron_params: Optional[Dict] = None,
        recurrent: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.recurrent = recurrent
        
        # Linear transformation
        self.linear = nn.Linear(in_features, out_features, bias=True)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        
        # Recurrent connections
        if recurrent:
            self.recurrent_weight = nn.Parameter(torch.randn(out_features, out_features) * 0.01)
        
        # Neuron model
        neuron_params = neuron_params or {}
        if neuron_type == "LIF":
            self.neuron = LIFNeuron(**neuron_params)
        elif neuron_type == "AdaptiveLIF":
            self.neuron = AdaptiveLIFNeuron(**neuron_params)
        elif neuron_type == "Izhikevich":
            self.neuron = IzhikevichNeuron(**neuron_params)
        else:
            raise ValueError(f"Unknown neuron type: {neuron_type}")
    
    def forward(
        self,
        x: torch.Tensor,  # [batch, time, features] or [batch, features]
        state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            state: Previous state
            
        Returns:
            spikes: Output spikes
            new_state: Updated state
        """
        # Handle both temporal and static inputs
        if x.dim() == 3:
            # Temporal input [batch, time, features]
            return self.forward_temporal(x, state)
        else:
            # Static input [batch, features]
            current = self.linear(x)
            if self.recurrent and state is not None:
                current = current + torch.matmul(state.get("spikes", torch.zeros_like(current)), self.recurrent_weight)
            spikes, new_state = self.neuron(current, state)
            new_state["spikes"] = spikes
            return spikes, new_state
    
    def forward_temporal(
        self,
        x: torch.Tensor,  # [batch, time, features]
        state: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Forward pass for temporal input"""
        batch_size, time_steps, _ = x.shape
        
        spikes_list = []
        for t in range(time_steps):
            current = self.linear(x[:, t, :])
            if self.recurrent and state is not None:
                current = current + torch.matmul(state.get("spikes", torch.zeros_like(current)), self.recurrent_weight)
            
            spike, state = self.neuron(current, state)
            state["spikes"] = spike
            spikes_list.append(spike)
        
        spikes = torch.stack(spikes_list, dim=1)  # [batch, time, neurons]
        return spikes, state
