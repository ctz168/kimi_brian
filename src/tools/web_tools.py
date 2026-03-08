"""
Web Tools and Skills
网页技能和工具调用

Provides tools for:
- Wikipedia search
- Web search
- Calculator
- Code execution
"""

import json
import re
import math
import requests
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
import traceback
import subprocess
import tempfile
import os

# Try importing optional dependencies
try:
    import wikipedia
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

try:
    from duckduckgo_search import DDGS
    DDG_AVAILABLE = True
except ImportError:
    DDG_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    result: Any
    tool_name: str
    execution_time: float
    error: Optional[str] = None


class WikipediaTool:
    """
    Wikipedia search tool
    维基百科搜索工具
    """
    
    def __init__(self, language: str = "zh", max_results: int = 5):
        self.language = language
        self.max_results = max_results
        
        if WIKIPEDIA_AVAILABLE:
            wikipedia.set_lang(language)
    
    def search(self, query: str, sentences: int = 5) -> ToolResult:
        """Search Wikipedia"""
        if not WIKIPEDIA_AVAILABLE:
            return ToolResult(
                success=False,
                result=None,
                tool_name="wikipedia",
                execution_time=0,
                error="Wikipedia module not installed. Install with: pip install wikipedia",
            )
        
        start_time = datetime.now()
        
        try:
            # Search for pages
            search_results = wikipedia.search(query, results=self.max_results)
            
            if not search_results:
                return ToolResult(
                    success=True,
                    result=[],
                    tool_name="wikipedia",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            
            # Get summaries
            results = []
            for title in search_results[:self.max_results]:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    summary = wikipedia.summary(title, sentences=sentences, auto_suggest=False)
                    
                    results.append({
                        "title": title,
                        "summary": summary,
                        "url": page.url,
                        "content": page.content[:2000] if len(page.content) > 2000 else page.content,
                    })
                except wikipedia.DisambiguationError as e:
                    # Handle disambiguation
                    results.append({
                        "title": title,
                        "summary": f"Disambiguation page. Options: {', '.join(e.options[:5])}",
                        "url": "",
                        "content": "",
                    })
                except Exception as e:
                    continue
            
            return ToolResult(
                success=True,
                result=results,
                tool_name="wikipedia",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                tool_name="wikipedia",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
    
    def get_page(self, title: str) -> ToolResult:
        """Get full Wikipedia page"""
        if not WIKIPEDIA_AVAILABLE:
            return ToolResult(
                success=False,
                result=None,
                tool_name="wikipedia",
                execution_time=0,
                error="Wikipedia module not installed.",
            )
        
        start_time = datetime.now()
        
        try:
            page = wikipedia.page(title, auto_suggest=False)
            
            return ToolResult(
                success=True,
                result={
                    "title": page.title,
                    "content": page.content,
                    "summary": page.summary,
                    "url": page.url,
                    "references": page.references[:10],
                    "links": page.links[:20],
                },
                tool_name="wikipedia",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                tool_name="wikipedia",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )


class WebSearchTool:
    """
    Web search tool using DuckDuckGo
    网页搜索工具
    """
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
    
    def search(self, query: str) -> ToolResult:
        """Search the web"""
        if not DDG_AVAILABLE:
            return ToolResult(
                success=False,
                result=None,
                tool_name="web_search",
                execution_time=0,
                error="DuckDuckGo search not installed. Install with: pip install duckduckgo-search",
            )
        
        start_time = datetime.now()
        
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self.max_results))
                
                formatted_results = [
                    {
                        "title": r.get("title", ""),
                        "body": r.get("body", ""),
                        "href": r.get("href", ""),
                    }
                    for r in results
                ]
                
                return ToolResult(
                    success=True,
                    result=formatted_results,
                    tool_name="web_search",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
                
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                tool_name="web_search",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )
    
    def fetch_page(self, url: str) -> ToolResult:
        """Fetch and parse a web page"""
        if not BS4_AVAILABLE:
            return ToolResult(
                success=False,
                result=None,
                tool_name="fetch_page",
                execution_time=0,
                error="BeautifulSoup not installed. Install with: pip install beautifulsoup4",
            )
        
        start_time = datetime.now()
        
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines[:100])  # Limit length
            
            return ToolResult(
                success=True,
                result={
                    "url": url,
                    "title": soup.title.string if soup.title else "",
                    "text": text,
                },
                tool_name="fetch_page",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                tool_name="fetch_page",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )


class CalculatorTool:
    """
    Calculator tool for mathematical expressions
    计算器工具
    """
    
    def __init__(self):
        # Safe math functions
        self.safe_dict = {
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'pow': pow,
            'round': round,
            'len': len,
            'math': math,
        }
        
        # Add math functions
        for name in dir(math):
            if not name.startswith('_'):
                self.safe_dict[name] = getattr(math, name)
    
    def calculate(self, expression: str) -> ToolResult:
        """Calculate mathematical expression"""
        start_time = datetime.now()
        
        try:
            # Clean expression
            expression = expression.strip()
            
            # Remove potentially dangerous characters
            cleaned = re.sub(r'[^\d\+\-\*\/\(\)\.\,\s\w\[\]\{\}]', '', expression)
            
            # Evaluate
            result = eval(cleaned, {"__builtins__": {}}, self.safe_dict)
            
            return ToolResult(
                success=True,
                result=result,
                tool_name="calculator",
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                tool_name="calculator",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )


class CodeInterpreterTool:
    """
    Code interpreter tool (safe execution)
    代码解释器工具
    
    WARNING: This executes code. Use with caution.
    """
    
    def __init__(self, timeout: int = 5):
        self.timeout = timeout
    
    def execute(self, code: str, language: str = "python") -> ToolResult:
        """Execute code safely"""
        start_time = datetime.now()
        
        if language != "python":
            return ToolResult(
                success=False,
                result=None,
                tool_name="code_interpreter",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=f"Language {language} not supported. Only Python is supported.",
            )
        
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute with timeout
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            # Clean up
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return ToolResult(
                    success=True,
                    result=result.stdout,
                    tool_name="code_interpreter",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                )
            else:
                return ToolResult(
                    success=False,
                    result=None,
                    tool_name="code_interpreter",
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    error=result.stderr,
                )
                
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                result=None,
                tool_name="code_interpreter",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=f"Code execution timed out after {self.timeout} seconds",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                tool_name="code_interpreter",
                execution_time=(datetime.now() - start_time).total_seconds(),
                error=str(e),
            )


class ToolRegistry:
    """
    Tool registry for managing and calling tools
    工具注册表
    """
    
    def __init__(self):
        self.tools: Dict[str, Any] = {}
        self.descriptions: Dict[str, Dict] = {}
        
        # Initialize default tools
        self._init_default_tools()
    
    def _init_default_tools(self):
        """Initialize default tools"""
        # Wikipedia
        wiki = WikipediaTool()
        self.register(
            name="wikipedia_search",
            tool=wiki,
            method="search",
            description="Search Wikipedia for information",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "sentences": {"type": "integer", "description": "Number of sentences in summary", "default": 5},
            },
        )
        
        # Web search
        web = WebSearchTool()
        self.register(
            name="web_search",
            tool=web,
            method="search",
            description="Search the web using DuckDuckGo",
            parameters={
                "query": {"type": "string", "description": "Search query"},
            },
        )
        
        # Calculator
        calc = CalculatorTool()
        self.register(
            name="calculate",
            tool=calc,
            method="calculate",
            description="Calculate mathematical expressions",
            parameters={
                "expression": {"type": "string", "description": "Mathematical expression to evaluate"},
            },
        )
        
        # Code interpreter
        code = CodeInterpreterTool()
        self.register(
            name="execute_code",
            tool=code,
            method="execute",
            description="Execute Python code safely",
            parameters={
                "code": {"type": "string", "description": "Python code to execute"},
                "language": {"type": "string", "description": "Programming language", "default": "python"},
            },
        )
    
    def register(
        self,
        name: str,
        tool: Any,
        method: str,
        description: str,
        parameters: Dict[str, Any],
    ):
        """Register a tool"""
        self.tools[name] = {
            "tool": tool,
            "method": method,
        }
        self.descriptions[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
        }
    
    def call(self, name: str, **kwargs) -> ToolResult:
        """Call a tool"""
        if name not in self.tools:
            return ToolResult(
                success=False,
                result=None,
                tool_name=name,
                execution_time=0,
                error=f"Tool '{name}' not found",
            )
        
        tool_info = self.tools[name]
        tool = tool_info["tool"]
        method = getattr(tool, tool_info["method"])
        
        return method(**kwargs)
    
    def get_available_tools(self) -> List[Dict]:
        """Get list of available tools"""
        return list(self.descriptions.values())
    
    def get_tool_description(self, name: str) -> Optional[Dict]:
        """Get description of a tool"""
        return self.descriptions.get(name)


class StreamingToolExecutor:
    """
    Streaming tool executor for high-refresh-rate processing
    流式工具执行器
    
    Executes tools while streaming LLM output.
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
        self.pending_calls: List[Dict] = []
        self.results: List[ToolResult] = []
    
    def detect_tool_calls(self, text: str) -> List[Dict]:
        """Detect tool calls in text"""
        # Look for patterns like [TOOL:tool_name]{...}
        pattern = r'\[TOOL:(\w+)\]\s*(\{[^\}]*\})'
        matches = re.findall(pattern, text)
        
        calls = []
        for tool_name, args_str in matches:
            try:
                args = json.loads(args_str)
                calls.append({
                    "tool": tool_name,
                    "args": args,
                })
            except json.JSONDecodeError:
                continue
        
        return calls
    
    def execute_pending(self) -> List[ToolResult]:
        """Execute pending tool calls"""
        results = []
        for call in self.pending_calls:
            result = self.tool_registry.call(call["tool"], **call["args"])
            results.append(result)
            self.results.append(result)
        
        self.pending_calls = []
        return results
    
    def format_results(self, results: List[ToolResult]) -> str:
        """Format tool results for LLM consumption"""
        formatted = []
        for result in results:
            if result.success:
                formatted.append(f"[TOOL_RESULT:{result.tool_name}]\n{json.dumps(result.result, ensure_ascii=False, indent=2)}")
            else:
                formatted.append(f"[TOOL_ERROR:{result.tool_name}]\n{result.error}")
        
        return "\n\n".join(formatted)
