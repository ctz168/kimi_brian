"""
Streaming Inference Module
流式推理模块

High-refresh-rate streaming inference with:
- Real-time token generation
- Memory retrieval during generation
- Tool calling integration
- Online learning during inference
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any, Generator, AsyncGenerator
import threading
import asyncio
from dataclasses import dataclass
from datetime import datetime
import time

from ..models.model_integration import BrainInspiredModel, StreamToken
from ..tools.web_tools import ToolRegistry, StreamingToolExecutor
from ..training.online_learning import StreamingLearner


@dataclass
class InferenceConfig:
    """Inference configuration"""
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    
    # Streaming settings
    refresh_rate: int = 60  # Hz
    stream_buffer_size: int = 10
    
    # Memory settings
    use_memory: bool = True
    memory_top_k: int = 5
    memory_refresh_interval: int = 50  # tokens
    
    # Tool settings
    use_tools: bool = True
    tool_detection_interval: int = 20  # tokens
    
    # Learning settings
    use_online_learning: bool = True
    learning_update_interval: int = 10


class StreamingInference:
    """
    High-refresh-rate streaming inference engine
    高刷新率流式推理引擎
    
    Features:
    - 60Hz+ token streaming
    - Real-time memory retrieval
    - Dynamic tool calling
    - Online learning integration
    """
    
    def __init__(
        self,
        model: BrainInspiredModel,
        tool_registry: Optional[ToolRegistry] = None,
        streaming_learner: Optional[StreamingLearner] = None,
        config: Optional[InferenceConfig] = None,
    ):
        self.model = model
        self.tool_registry = tool_registry or ToolRegistry()
        self.streaming_learner = streaming_learner
        self.config = config or InferenceConfig()
        
        # Tool executor
        self.tool_executor = StreamingToolExecutor(self.tool_registry)
        
        # Streaming state
        self.is_generating = False
        self.generation_lock = threading.Lock()
        
        # Performance metrics
        self.metrics = {
            "tokens_generated": 0,
            "total_generation_time": 0.0,
            "memory_retrieval_time": 0.0,
            "tool_execution_time": 0.0,
        }
    
    def generate(
        self,
        prompt: str,
        images: Optional[List] = None,
        system_prompt: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ) -> Generator[StreamToken, None, None]:
        """
        Stream generation with high refresh rate
        
        Args:
            prompt: User prompt
            images: Optional images
            system_prompt: Optional system prompt
            config: Optional override config
            
        Yields:
            StreamToken objects
        """
        config = config or self.config
        
        with self.generation_lock:
            self.is_generating = True
            
            try:
                # Build full prompt
                full_prompt = self._build_prompt(prompt, system_prompt)
                
                # Initial memory retrieval
                memories = []
                if config.use_memory:
                    memories = self._retrieve_memories(full_prompt)
                    if memories:
                        yield StreamToken(
                            token="",
                            token_id=0,
                            is_special=True,
                            memory_context=memories,
                        )
                
                # Token generation
                token_buffer = []
                token_count = 0
                
                start_time = time.time()
                
                for token in self.model.generate_stream(
                    prompt=full_prompt,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    use_memory=config.use_memory,
                    use_snn=True,
                    images=images,
                ):
                    if not self.is_generating:
                        break
                    
                    token_count += 1
                    token_buffer.append(token)
                    
                    # Yield token immediately for high refresh rate
                    yield token
                    
                    # Periodic memory refresh
                    if config.use_memory and token_count % config.memory_refresh_interval == 0:
                        context = "".join([t.token for t in token_buffer[-20:]])
                        new_memories = self._retrieve_memories(context)
                        if new_memories:
                            yield StreamToken(
                                token="",
                                token_id=0,
                                is_special=True,
                                memory_context=new_memories,
                            )
                    
                    # Tool detection and execution
                    if config.use_tools and token_count % config.tool_detection_interval == 0:
                        context = "".join([t.token for t in token_buffer[-50:]])
                        tool_calls = self.tool_executor.detect_tool_calls(context)
                        
                        if tool_calls:
                            self.tool_executor.pending_calls = tool_calls
                            results = self.tool_executor.execute_pending()
                            
                            # Yield tool results
                            for result in results:
                                yield StreamToken(
                                    token=f"\n[Tool: {result.tool_name}]\n{result.result}\n",
                                    token_id=0,
                                    is_special=True,
                                )
                    
                    # Online learning update
                    if config.use_online_learning and self.streaming_learner:
                        self.streaming_learner.on_token_generated(
                            token=token.token,
                            hidden_state=token.hidden_state if token.hidden_state else torch.zeros(1, 768),
                            memory_context=memories if token_count == 1 else None,
                        )
                    
                    # Maintain buffer size
                    if len(token_buffer) > config.stream_buffer_size:
                        token_buffer.pop(0)
                
                # Finalize
                generation_time = time.time() - start_time
                self.metrics["tokens_generated"] += token_count
                self.metrics["total_generation_time"] += generation_time
                
                # Sequence complete callback
                if config.use_online_learning and self.streaming_learner:
                    full_response = "".join([t.token for t in token_buffer])
                    self.streaming_learner.on_sequence_complete(full_response)
                
            finally:
                self.is_generating = False
    
    async def generate_async(
        self,
        prompt: str,
        images: Optional[List] = None,
        system_prompt: Optional[str] = None,
        config: Optional[InferenceConfig] = None,
    ) -> AsyncGenerator[StreamToken, None]:
        """Async streaming generation"""
        config = config or self.config
        
        loop = asyncio.get_event_loop()
        
        # Run sync generator in executor
        def sync_generate():
            return list(self.generate(prompt, images, system_prompt, config))
        
        tokens = await loop.run_in_executor(None, sync_generate)
        
        for token in tokens:
            yield token
            await asyncio.sleep(0)  # Allow other tasks
    
    def _build_prompt(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Build full prompt with system message"""
        if system_prompt:
            return f"<|system|>\n{system_prompt}\n<|user|>\n{user_prompt}\n<|assistant|>\n"
        else:
            return f"<|user|>\n{user_prompt}\n<|assistant|>\n"
    
    def _retrieve_memories(self, query: str) -> List[Dict]:
        """Retrieve relevant memories"""
        start_time = time.time()
        
        # Encode query
        query_embedding = self.model.encode_multimodal(text=query)
        
        # Retrieve from model's memory
        memories = self.model.retrieve_memory(
            query_embedding,
            top_k=self.config.memory_top_k,
        )
        
        retrieval_time = time.time() - start_time
        self.metrics["memory_retrieval_time"] += retrieval_time
        
        return memories
    
    def stop_generation(self):
        """Stop ongoing generation"""
        self.is_generating = False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = self.metrics.copy()
        
        if metrics["tokens_generated"] > 0:
            metrics["tokens_per_second"] = (
                metrics["tokens_generated"] / metrics["total_generation_time"]
            )
        else:
            metrics["tokens_per_second"] = 0.0
        
        return metrics


class BatchInference:
    """
    Batch inference for processing multiple inputs
    批量推理
    """
    
    def __init__(
        self,
        model: BrainInspiredModel,
        batch_size: int = 8,
    ):
        self.model = model
        self.batch_size = batch_size
    
    def process_batch(
        self,
        prompts: List[str],
        **generation_kwargs,
    ) -> List[str]:
        """Process batch of prompts"""
        results = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            
            # Process each prompt in batch
            batch_results = []
            for prompt in batch:
                response = self.model.generate(prompt, **generation_kwargs)
                batch_results.append(response)
            
            results.extend(batch_results)
        
        return results


class InteractiveSession:
    """
    Interactive session with context management
    交互式会话
    """
    
    def __init__(
        self,
        inference_engine: StreamingInference,
        max_context_length: int = 4096,
    ):
        self.inference_engine = inference_engine
        self.max_context_length = max_context_length
        
        # Conversation history
        self.history: List[Dict[str, str]] = []
        self.session_memory: List[Dict] = []
    
    def chat(
        self,
        message: str,
        images: Optional[List] = None,
    ) -> Generator[StreamToken, None, None]:
        """
        Send a message and get streaming response
        
        Args:
            message: User message
            images: Optional images
            
        Yields:
            StreamToken objects
        """
        # Build context from history
        context = self._build_context()
        
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Generate response
        full_response = []
        
        for token in self.inference_engine.generate(
            prompt=message,
            images=images,
            system_prompt=context,
        ):
            yield token
            full_response.append(token.token)
        
        # Add assistant response to history
        response_text = "".join(full_response)
        self.history.append({"role": "assistant", "content": response_text})
        
        # Trim history if too long
        self._trim_history()
    
    def _build_context(self) -> str:
        """Build context from conversation history"""
        if not self.history:
            return ""
        
        context_parts = []
        for turn in self.history[-10:]:  # Last 10 turns
            role = turn["role"]
            content = turn["content"]
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    def _trim_history(self):
        """Trim history to fit within context length"""
        # Simple token estimation
        total_length = sum(len(turn["content"]) for turn in self.history)
        
        while total_length > self.max_context_length and len(self.history) > 2:
            # Remove oldest turn
            removed = self.history.pop(0)
            total_length -= len(removed["content"])
    
    def clear_history(self):
        """Clear conversation history"""
        self.history.clear()
        self.session_memory.clear()
    
    def get_summary(self) -> str:
        """Get session summary"""
        return f"Session has {len(self.history)} turns, {len(self.session_memory)} memories"


class InferencePipeline:
    """
    Complete inference pipeline with all features
    完整推理管道
    """
    
    def __init__(
        self,
        model: BrainInspiredModel,
        tool_registry: Optional[ToolRegistry] = None,
        enable_streaming: bool = True,
        enable_learning: bool = True,
    ):
        self.model = model
        
        # Create streaming learner if enabled
        streaming_learner = None
        if enable_learning:
            from ..training.online_learning import OnlineLearner
            online_learner = OnlineLearner(model)
            streaming_learner = StreamingLearner(online_learner)
        
        # Create inference engine
        self.inference_engine = StreamingInference(
            model=model,
            tool_registry=tool_registry,
            streaming_learner=streaming_learner,
        )
        
        # Session management
        self.sessions: Dict[str, InteractiveSession] = {}
    
    def create_session(self, session_id: str) -> InteractiveSession:
        """Create new interactive session"""
        session = InteractiveSession(self.inference_engine)
        self.sessions[session_id] = session
        return session
    
    def get_session(self, session_id: str) -> Optional[InteractiveSession]:
        """Get existing session"""
        return self.sessions.get(session_id)
    
    def close_session(self, session_id: str):
        """Close and cleanup session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def generate(
        self,
        prompt: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Generator[StreamToken, None, None]:
        """Generate with optional session context"""
        if session_id and session_id in self.sessions:
            session = self.sessions[session_id]
            yield from session.chat(prompt, **kwargs)
        else:
            yield from self.inference_engine.generate(prompt, **kwargs)
