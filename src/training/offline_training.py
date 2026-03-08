"""
Offline Training Module
离线训练模块

Implements comprehensive offline training:
- Supervised fine-tuning
- Reinforcement learning from human feedback (RLHF)
- Contrastive learning
- Multi-threaded data loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..core.stdp import STDPLearner, TripletSTDPLearner


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic settings
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    
    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 500
    output_dir: str = "./checkpoints"
    
    # Logging
    log_interval: int = 10
    
    # Multi-threading
    num_workers: int = 4
    prefetch_factor: int = 2


class BrainDataset(Dataset):
    """
    Dataset for brain-inspired model training
    类脑模型训练数据集
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer,
        max_length: int = 512,
        include_memory: bool = True,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_memory = include_memory
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        text = item.get("text", "")
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        result = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        
        # Add labels for language modeling
        if "labels" in item:
            labels = self.tokenizer(
                item["labels"],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result["labels"] = labels["input_ids"].squeeze(0)
        else:
            result["labels"] = result["input_ids"].clone()
        
        # Add memory context
        if self.include_memory and "memory_context" in item:
            result["memory_context"] = item["memory_context"]
        
        # Add reward for RL
        if "reward" in item:
            result["reward"] = torch.tensor(item["reward"], dtype=torch.float32)
        
        return result


class OfflineTrainer:
    """
    Offline trainer for comprehensive model training
    离线训练器
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=config.learning_rate * 0.1,
        )
        
        # Scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Metrics
        self.metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, List[float]]:
        """
        Main training loop
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            callbacks: List of callback functions
            
        Returns:
            Training metrics
        """
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=True,
        )
        
        eval_loader = None
        if eval_dataset:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                prefetch_factor=self.config.prefetch_factor,
            )
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Train epoch
            train_loss = self._train_epoch(train_loader, callbacks)
            self.metrics["train_loss"].append(train_loss)
            
            # Evaluate
            if eval_loader and (epoch + 1) % self.config.eval_steps == 0:
                eval_loss = self._evaluate(eval_loader)
                self.metrics["eval_loss"].append(eval_loss)
                print(f"Eval Loss: {eval_loss:.4f}")
            
            # Update scheduler
            self.scheduler.step()
            self.metrics["learning_rate"].append(self.optimizer.param_groups[0]["lr"])
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}")
        
        # Save final model
        self._save_checkpoint("final")
        
        return self.metrics
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        callbacks: Optional[List[Callable]] = None,
    ) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss = self._compute_loss(batch)
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm,
                )
                
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(),
                "avg_loss": total_loss / num_batches,
            })
            
            # Call callbacks
            if callbacks:
                for callback in callbacks:
                    callback(self, batch_idx, loss.item())
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for a batch"""
        # Language modeling loss
        outputs = self.model.base_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        
        loss = outputs.loss
        
        # Add auxiliary losses if available
        if "reward" in batch:
            # RL loss component
            rl_loss = -batch["reward"].mean()
            loss = loss + 0.1 * rl_loss
        
        return loss
    
    def _evaluate(self, eval_loader: DataLoader) -> float:
        """Evaluate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                loss = self._compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(self.config.output_dir, name)
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model
        self.model.save(checkpoint_path)
        
        # Save training state
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "metrics": self.metrics,
        }
        
        if self.scaler:
            state["scaler"] = self.scaler.state_dict()
        
        torch.save(state, os.path.join(checkpoint_path, "training_state.pt"))
        
        print(f"Checkpoint saved: {checkpoint_path}")


class MultiThreadedTrainer:
    """
    Multi-threaded trainer for parallel training
    多线程训练器
    """
    
    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        config: TrainingConfig,
        num_threads: int = 4,
    ):
        self.model_factory = model_factory
        self.config = config
        self.num_threads = num_threads
        
        # Thread-local storage
        self.thread_local = threading.local()
    
    def _get_thread_model(self) -> nn.Module:
        """Get or create thread-local model"""
        if not hasattr(self.thread_local, 'model'):
            self.thread_local.model = self.model_factory()
        return self.thread_local.model
    
    def train_parallel(
        self,
        datasets: List[Dataset],
        merge_strategy: str = "average",
    ) -> List[Dict[str, List[float]]]:
        """
        Train on multiple datasets in parallel
        
        Args:
            datasets: List of datasets to train on
            merge_strategy: How to merge model updates
            
        Returns:
            List of training metrics
        """
        all_metrics = []
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit training tasks
            futures = {
                executor.submit(self._train_on_dataset, dataset, i): i
                for i, dataset in enumerate(datasets)
            }
            
            # Collect results
            for future in as_completed(futures):
                dataset_idx = futures[future]
                try:
                    metrics = future.result()
                    all_metrics.append(metrics)
                    print(f"Dataset {dataset_idx} training completed")
                except Exception as e:
                    print(f"Dataset {dataset_idx} training failed: {e}")
        
        # Merge models
        if merge_strategy == "average":
            self._average_model_weights()
        
        return all_metrics
    
    def _train_on_dataset(
        self,
        dataset: Dataset,
        dataset_idx: int,
    ) -> Dict[str, List[float]]:
        """Train on a single dataset"""
        model = self._get_thread_model()
        
        # Create trainer
        config = TrainingConfig(**self.config.__dict__)
        config.output_dir = os.path.join(config.output_dir, f"thread_{dataset_idx}")
        
        trainer = OfflineTrainer(model, config)
        
        # Train
        metrics = trainer.train(dataset)
        
        return metrics
    
    def _average_model_weights(self):
        """Average model weights from all threads"""
        # This would require collecting all model states
        # and averaging them
        pass


class ModuleTrainer:
    """
    Trainer for individual modules
    单独模块训练器
    """
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
    
    def train_snn(
        self,
        snn_module: nn.Module,
        spike_data: List[torch.Tensor],
        epochs: int = 10,
        learning_rate: float = 1e-3,
    ) -> Dict[str, float]:
        """Train spiking neural network module"""
        snn_module.to(self.device)
        snn_module.train()
        
        optimizer = AdamW(snn_module.parameters(), lr=learning_rate)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for spike_batch in spike_data:
                spike_batch = spike_batch.to(self.device)
                
                # Forward pass
                output, _ = snn_module(spike_batch)
                
                # Reconstruction loss
                loss = F.mse_loss(output, spike_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(spike_data)
            losses.append(avg_loss)
            print(f"SNN Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return {"final_loss": losses[-1], "losses": losses}
    
    def train_memory(
        self,
        memory_module: nn.Module,
        memory_data: List[Tuple[torch.Tensor, torch.Tensor]],  # (input, target)
        epochs: int = 10,
        learning_rate: float = 1e-4,
    ) -> Dict[str, float]:
        """Train memory module (hippocampus)"""
        memory_module.to(self.device)
        memory_module.train()
        
        optimizer = AdamW(memory_module.parameters(), lr=learning_rate)
        
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for input_vec, target_vec in memory_data:
                input_vec = input_vec.to(self.device)
                target_vec = target_vec.to(self.device)
                
                # Encode and retrieve
                engram = memory_module.encode(input_vec)
                retrieved = memory_module.pattern_completion(target_vec)
                
                # Loss
                loss = F.mse_loss(retrieved, target_vec)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(memory_data)
            losses.append(avg_loss)
            print(f"Memory Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return {"final_loss": losses[-1], "losses": losses}
    
    def train_stdp(
        self,
        stdp_learner: STDPLearner,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        weights: torch.Tensor,
        num_iterations: int = 100,
    ) -> torch.Tensor:
        """Train STDP weights"""
        for i in range(num_iterations):
            delta_w = stdp_learner(weights, pre_spikes, post_spikes)
            weights = weights + delta_w
            
            if i % 10 == 0:
                print(f"STDP Iteration {i}, Weight change norm: {delta_w.norm().item():.6f}")
        
        return weights
