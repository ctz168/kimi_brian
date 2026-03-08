"""
Streamlit Web Interface for Brain-Inspired AI
Streamlit网页界面

Features:
- Interactive chat interface
- Real-time streaming display
- Memory visualization
- Tool usage display
- Model configuration
"""

import streamlit as st
import requests
import json
import asyncio
import websockets
from typing import List, Dict, Optional
from datetime import datetime
import threading
import queue

# API configuration
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/chat"


# Page configuration
st.set_page_config(
    page_title="Brain-Inspired AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    .assistant-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .memory-badge {
        background-color: #FFF3E0;
        border: 1px solid #FF9800;
        border-radius: 0.25rem;
        padding: 0.25rem 0.5rem;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
    }
    .tool-result {
        background-color: #E8F5E9;
        border: 1px solid #4CAF50;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# Session state initialization
def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    
    if "streaming" not in st.session_state:
        st.session_state.streaming = False
    
    if "memories" not in st.session_state:
        st.session_state.memories = []
    
    if "tools_used" not in st.session_state:
        st.session_state.tools_used = []
    
    if "model_config" not in st.session_state:
        st.session_state.model_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 512,
            "use_memory": True,
            "use_tools": True,
        }


# API functions
def check_api_status() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info() -> Optional[Dict]:
    """Get model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def generate_text(prompt: str, system_prompt: str = "", stream: bool = True) -> str:
    """Generate text using API"""
    config = st.session_state.model_config
    
    payload = {
        "prompt": prompt,
        "system_prompt": system_prompt if system_prompt else None,
        "max_new_tokens": config["max_tokens"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "top_k": config["top_k"],
        "use_memory": config["use_memory"],
        "use_tools": config["use_tools"],
    }
    
    try:
        if stream:
            response = requests.post(
                f"{API_BASE_URL}/generate/stream",
                json=payload,
                stream=True,
                timeout=60,
            )
            
            full_text = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        
                        if 'error' in data:
                            st.error(f"Error: {data['error']}")
                            break
                        
                        if 'done' in data:
                            break
                        
                        if 'token' in data:
                            full_text += data['token']
                            
                        if 'memories' in data:
                            st.session_state.memories = data['memories']
            
            return full_text
        else:
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json=payload,
                timeout=60,
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'memories_used' in result:
                    st.session_state.memories = result['memories_used']
                return result.get('text', '')
            else:
                st.error(f"API Error: {response.text}")
                return ""
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return ""


def query_memory(query: str, top_k: int = 5) -> List[Dict]:
    """Query memory system"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/memory/query",
            json={"query": query, "top_k": top_k},
            timeout=10,
        )
        
        if response.status_code == 200:
            return response.json()
        return []
    
    except:
        return []


def store_memory(content: str, memory_type: str = "episodic"):
    """Store content in memory"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/memory/store",
            params={"content": content, "memory_type": memory_type},
            timeout=10,
        )
        return response.status_code == 200
    except:
        return False


def get_available_tools() -> List[Dict]:
    """Get list of available tools"""
    try:
        response = requests.get(f"{API_BASE_URL}/tools/list", timeout=5)
        if response.status_code == 200:
            return response.json()
        return []
    except:
        return []


def get_metrics() -> Dict:
    """Get performance metrics"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


# UI Components
def render_sidebar():
    """Render sidebar with configuration"""
    with st.sidebar:
        st.markdown("### 🧠 Brain-Inspired AI")
        
        # API Status
        api_status = check_api_status()
        if api_status:
            st.success("✅ API Connected")
        else:
            st.error("❌ API Disconnected")
            st.info("Start the API server with: `python -m src.api.fastapi_server`")
        
        st.divider()
        
        # Model Configuration
        st.markdown("### ⚙️ Configuration")
        
        st.session_state.model_config["temperature"] = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
        )
        
        st.session_state.model_config["top_p"] = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
        )
        
        st.session_state.model_config["max_tokens"] = st.slider(
            "Max Tokens",
            min_value=64,
            max_value=2048,
            value=512,
            step=64,
        )
        
        st.session_state.model_config["use_memory"] = st.toggle(
            "Use Memory",
            value=True,
        )
        
        st.session_state.model_config["use_tools"] = st.toggle(
            "Use Tools",
            value=True,
        )
        
        st.divider()
        
        # System Prompt
        st.markdown("### 📝 System Prompt")
        system_prompt = st.text_area(
            "System Instructions",
            value="You are a helpful AI assistant with brain-inspired capabilities including memory and learning.",
            height=100,
        )
        
        st.divider()
        
        # Actions
        st.markdown("### 🎯 Actions")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memories = []
            st.rerun()
        
        if st.button("💾 Save Session", use_container_width=True):
            st.info("Session saved!")
        
        st.divider()
        
        # Model Info
        model_info = get_model_info()
        if model_info:
            st.markdown("### 📊 Model Info")
            st.json(model_info)


def render_chat_interface():
    """Render main chat interface"""
    st.markdown('<h1 class="main-header">🧠 Brain-Inspired AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">High-refresh-rate neural computing with online learning</p>', 
                unsafe_allow_html=True)
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><strong>👤 You:</strong><br>{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message"><strong>🤖 AI:</strong><br>{content}</div>',
                    unsafe_allow_html=True,
                )
                
                # Display memories used
                if "memories" in message and message["memories"]:
                    st.markdown("<small>💭 <strong>Memories used:</strong></small>", unsafe_allow_html=True)
                    for mem in message["memories"][:3]:
                        st.markdown(
                            f'<span class="memory-badge">{mem.get("content", "")[:50]}...</span>',
                            unsafe_allow_html=True,
                        )
    
    # Input area
    st.divider()
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Type your message here...",
            key="user_input",
            label_visibility="collapsed",
        )
    
    with col2:
        send_button = st.button("➤ Send", use_container_width=True)
    
    # Handle send
    if send_button and user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Generate response
        with st.spinner("Thinking..."):
            response = generate_text(user_input)
        
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "memories": st.session_state.memories,
        })
        
        # Clear input and rerun
        st.rerun()


def render_memory_tab():
    """Render memory management tab"""
    st.markdown("### 🧠 Memory Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Query Memory")
        query = st.text_input("Search query")
        top_k = st.slider("Results", 1, 20, 5)
        
        if st.button("Search"):
            memories = query_memory(query, top_k)
            
            if memories:
                for mem in memories:
                    with st.expander(f"📄 {mem.get('content', '')[:50]}... (Score: {mem.get('score', 0):.3f})"):
                        st.write(f"**Content:** {mem.get('content', '')}")
                        st.write(f"**Type:** {mem.get('memory_type', 'unknown')}")
                        st.write(f"**Timestamp:** {mem.get('timestamp', '')}")
            else:
                st.info("No memories found")
    
    with col2:
        st.markdown("#### 💾 Store Memory")
        new_memory = st.text_area("Content to store")
        memory_type = st.selectbox(
            "Memory Type",
            ["episodic", "semantic", "procedural"],
        )
        
        if st.button("Store"):
            if store_memory(new_memory, memory_type):
                st.success("Memory stored successfully!")
            else:
                st.error("Failed to store memory")


def render_tools_tab():
    """Render tools tab"""
    st.markdown("### 🛠️ Tools")
    
    tools = get_available_tools()
    
    if tools:
        for tool in tools:
            with st.expander(f"🔧 {tool['name']}"):
                st.write(f"**Description:** {tool['description']}")
                st.write("**Parameters:**")
                for param_name, param_info in tool.get('parameters', {}).items():
                    st.write(f"- `{param_name}`: {param_info.get('description', '')}")
    else:
        st.info("No tools available")


def render_metrics_tab():
    """Render metrics tab"""
    st.markdown("### 📊 Performance Metrics")
    
    metrics = get_metrics()
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Tokens Generated",
                metrics.get("tokens_generated", 0),
            )
        
        with col2:
            st.metric(
                "Tokens/Second",
                f"{metrics.get('tokens_per_second', 0):.2f}",
            )
        
        with col3:
            st.metric(
                "Total Time",
                f"{metrics.get('total_generation_time', 0):.2f}s",
            )
        
        st.divider()
        
        # Detailed metrics
        st.json(metrics)
    else:
        st.info("No metrics available")


def render_about_tab():
    """Render about tab"""
    st.markdown("### 📖 About Brain-Inspired AI")
    
    st.markdown("""
    This is a **brain-inspired artificial intelligence system** featuring:
    
    #### 🧠 Core Features
    - **High Refresh Rate**: 60Hz+ streaming token generation
    - **Memory-Augmented**: Hippocampus-like episodic and semantic memory
    - **Online Learning**: Real-time STDP-based weight updates
    - **Multimodal**: Text, image, and audio processing
    - **Tool Integration**: Wikipedia, web search, calculator, code execution
    
    #### ⚡ Architecture
    - **Base Model**: Qwen2.5-0.5B-Instruct
    - **Spiking Neural Networks**: LIF and Adaptive LIF neurons
    - **Memory System**: Hippocampal memory with pattern completion
    - **Inference Engine**: Streaming with real-time memory retrieval
    
    #### 🔬 Learning Mechanisms
    - **STDP**: Spike-Timing-Dependent Plasticity
    - **Reward Modulation**: R-STDP for reinforcement learning
    - **Experience Replay**: Consolidation of recent experiences
    - **Neurogenesis**: Dynamic memory allocation
    
    #### 🛠️ Development
    - **GitHub**: [brain-inspired-ai](https://github.com/yourusername/brain-inspired-ai)
    - **Documentation**: See README.md
    - **License**: MIT
    """)


# Main app
def main():
    """Main application"""
    init_session_state()
    
    render_sidebar()
    
    # Main tabs
    tabs = st.tabs([
        "💬 Chat",
        "🧠 Memory",
        "🛠️ Tools",
        "📊 Metrics",
        "📖 About",
    ])
    
    with tabs[0]:
        render_chat_interface()
    
    with tabs[1]:
        render_memory_tab()
    
    with tabs[2]:
        render_tools_tab()
    
    with tabs[3]:
        render_metrics_tab()
    
    with tabs[4]:
        render_about_tab()


if __name__ == "__main__":
    main()
