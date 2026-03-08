"""
FastAPI Server for Brain-Inspired AI
FastAPI服务器

Provides RESTful API and WebSocket endpoints for:
- Text generation
- Multimodal processing
- Memory management
- Tool calling
"""

import asyncio
import json
import uuid
from typing import List, Dict, Optional, Any, AsyncGenerator
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
from ..models.model_integration import BrainInspiredModel, StreamToken
from ..inference.streaming_inference import (
    StreamingInference, 
    InferenceConfig, 
    InferencePipeline,
    InteractiveSession,
)
from ..tools.web_tools import ToolRegistry
from ..training.online_learning import OnlineLearner, StreamingLearner


# Pydantic models for API
class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=100)
    use_memory: bool = True
    use_tools: bool = True
    system_prompt: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    generation_time: float
    memories_used: List[Dict] = []


class ChatMessage(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    max_new_tokens: int = 512
    temperature: float = 0.7


class MemoryQueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    memory_type: Optional[str] = None


class MemoryEntryResponse(BaseModel):
    id: str
    content: str
    memory_type: str
    timestamp: datetime
    score: float


class ToolCallRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]


class ToolCallResponse(BaseModel):
    success: bool
    result: Any
    execution_time: float
    error: Optional[str] = None


class ModelInfoResponse(BaseModel):
    name: str
    version: str
    capabilities: List[str]
    parameters: Dict[str, Any]


# Global state
class AppState:
    def __init__(self):
        self.model: Optional[BrainInspiredModel] = None
        self.inference_engine: Optional[StreamingInference] = None
        self.pipeline: Optional[InferencePipeline] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.sessions: Dict[str, InteractiveSession] = {}


app_state = AppState()


# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    print("Starting Brain-Inspired AI Server...")
    
    # Initialize model
    try:
        app_state.model = BrainInspiredModel(
            base_model_name="Qwen/Qwen2.5-0.5B-Instruct",
            device="auto",
        )
        
        # Initialize tool registry
        app_state.tool_registry = ToolRegistry()
        
        # Initialize inference pipeline
        app_state.pipeline = InferencePipeline(
            model=app_state.model,
            tool_registry=app_state.tool_registry,
        )
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        # Continue without model for testing
        pass
    
    yield
    
    # Shutdown
    print("Shutting down...")
    if app_state.model:
        # Save state
        pass


# Create FastAPI app
app = FastAPI(
    title="Brain-Inspired AI API",
    description="High-refresh-rate brain-inspired AI with online learning",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Helper functions
def stream_tokens_to_string(tokens: List[StreamToken]) -> str:
    """Convert stream tokens to string"""
    return "".join([t.token for t in tokens if not t.is_special])


async def token_generator(prompt: str, **kwargs) -> AsyncGenerator[str, None]:
    """Generate tokens as SSE stream"""
    if not app_state.pipeline:
        yield f"data: {json.dumps({'error': 'Model not loaded'})}\n\n"
        return
    
    try:
        for token in app_state.pipeline.generate(prompt, **kwargs):
            data = {
                "token": token.token,
                "token_id": token.token_id,
                "is_special": token.is_special,
            }
            
            if token.memory_context:
                data["memories"] = token.memory_context
            
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming
        
        yield f"data: {json.dumps({'done': True})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Brain-Inspired AI API",
        "version": "1.0.0",
        "status": "running" if app_state.model else "model_not_loaded",
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    """Get model information"""
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return ModelInfoResponse(
        name="Brain-Inspired AI",
        version="1.0.0",
        capabilities=[
            "text_generation",
            "multimodal_processing",
            "memory_retrieval",
            "tool_calling",
            "online_learning",
            "streaming",
        ],
        parameters={
            "base_model": "Qwen2.5-0.5B-Instruct",
            "use_snn": True,
            "use_memory": True,
        },
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Non-streaming text generation"""
    if not app_state.pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        start_time = datetime.now()
        
        # Collect all tokens
        tokens = []
        memories = []
        
        for token in app_state.pipeline.generate(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
        ):
            tokens.append(token)
            if token.memory_context:
                memories.extend(token.memory_context)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return GenerateResponse(
            text=stream_tokens_to_string(tokens),
            tokens_generated=len([t for t in tokens if not t.is_special]),
            generation_time=generation_time,
            memories_used=memories[:5],
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Streaming text generation"""
    if not app_state.pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return StreamingResponse(
        token_generator(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
        ),
        media_type="text/event-stream",
    )


@app.post("/chat", response_model=GenerateResponse)
async def chat(request: ChatRequest):
    """Chat with conversation history"""
    if not app_state.pipeline:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get or create session
        session_id = request.session_id or str(uuid.uuid4())
        
        if session_id not in app_state.pipeline.sessions:
            app_state.pipeline.create_session(session_id)
        
        # Build prompt from messages
        last_message = request.messages[-1].content if request.messages else ""
        
        start_time = datetime.now()
        
        tokens = []
        for token in app_state.pipeline.generate(
            prompt=last_message,
            session_id=session_id,
        ):
            tokens.append(token)
        
        generation_time = (datetime.now() - start_time).total_seconds()
        
        return GenerateResponse(
            text=stream_tokens_to_string(tokens),
            tokens_generated=len([t for t in tokens if not t.is_special]),
            generation_time=generation_time,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket chat endpoint"""
    await websocket.accept()
    
    session_id = str(uuid.uuid4())
    if app_state.pipeline:
        app_state.pipeline.create_session(session_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message = data.get("message", "")
            
            if not app_state.pipeline:
                await websocket.send_json({"error": "Model not loaded"})
                continue
            
            # Stream response
            for token in app_state.pipeline.generate(
                prompt=message,
                session_id=session_id,
            ):
                await websocket.send_json({
                    "token": token.token,
                    "is_special": token.is_special,
                })
            
            await websocket.send_json({"done": True})
            
    except WebSocketDisconnect:
        print(f"Client disconnected: {session_id}")
        if app_state.pipeline:
            app_state.pipeline.close_session(session_id)


@app.post("/memory/query", response_model=List[MemoryEntryResponse])
async def query_memory(request: MemoryQueryRequest):
    """Query memory system"""
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode query
        query_embedding = app_state.model.encode_multimodal(text=request.query)
        
        # Retrieve memories
        memories = app_state.model.retrieve_memory(
            query_embedding,
            top_k=request.top_k,
        )
        
        return [
            MemoryEntryResponse(
                id=m.get("id", str(i)),
                content=m.get("content", ""),
                memory_type=m.get("type", "unknown"),
                timestamp=datetime.now(),
                score=m.get("score", 0.0),
            )
            for i, m in enumerate(memories)
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/store")
async def store_memory(content: str, memory_type: str = "episodic"):
    """Store content in memory"""
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Encode and store
        embedding = app_state.model.encode_multimodal(text=content)
        app_state.model.store_memory(content, embedding, memory_type)
        
        return {"success": True, "message": "Memory stored"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tool/call", response_model=ToolCallResponse)
async def call_tool(request: ToolCallRequest):
    """Call a tool"""
    if not app_state.tool_registry:
        raise HTTPException(status_code=503, detail="Tools not available")
    
    try:
        result = app_state.tool_registry.call(
            request.tool_name,
            **request.parameters,
        )
        
        return ToolCallResponse(
            success=result.success,
            result=result.result,
            execution_time=result.execution_time,
            error=result.error,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tools/list")
async def list_tools():
    """List available tools"""
    if not app_state.tool_registry:
        raise HTTPException(status_code=503, detail="Tools not available")
    
    return app_state.tool_registry.get_available_tools()


@app.post("/multimodal/process")
async def process_multimodal(
    text: Optional[str] = None,
    images: Optional[List[UploadFile]] = File(None),
):
    """Process multimodal input"""
    if not app_state.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Process images
        image_objects = []
        if images:
            from PIL import Image
            import io
            
            for image_file in images:
                contents = await image_file.read()
                image = Image.open(io.BytesIO(contents))
                image_objects.append(image)
        
        # Encode
        embedding = app_state.model.encode_multimodal(
            text=text,
            images=image_objects if image_objects else None,
        )
        
        return {
            "success": True,
            "embedding_shape": list(embedding.shape),
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": app_state.model is not None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    if not app_state.pipeline:
        return {"error": "Pipeline not initialized"}
    
    return app_state.pipeline.inference_engine.get_metrics()


# Main entry point
def start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
):
    """Start the API server"""
    uvicorn.run(
        "src.api.fastapi_server:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    start_server()
