"""
Benchmark and Evaluation Module
基准测试和评估模块

Comprehensive evaluation of brain-inspired AI:
- Language understanding (MMLU, C-Eval)
- Reasoning ability
- Memory performance
- Inference speed
- Comparison with baselines
"""

import torch
import time
import json
import os
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
from tqdm import tqdm
import requests


@dataclass
class BenchmarkResult:
    """Benchmark result"""
    benchmark_name: str
    score: float
    max_score: float
    metrics: Dict[str, float]
    timestamp: datetime
    details: List[Dict] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []
    
    def to_dict(self) -> Dict:
        return {
            "benchmark_name": self.benchmark_name,
            "score": self.score,
            "max_score": self.max_score,
            "percentage": (self.score / self.max_score * 100) if self.max_score > 0 else 0,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details,
        }


class LanguageUnderstandingBenchmark:
    """
    Language understanding benchmark
    语言理解基准测试
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        # Test datasets (simplified versions)
        self.test_questions = self._load_test_questions()
    
    def _load_test_questions(self) -> List[Dict]:
        """Load test questions"""
        # Sample questions covering various domains
        return [
            {
                "question": "What is the capital of France?",
                "choices": ["London", "Berlin", "Paris", "Madrid"],
                "answer": 2,
                "domain": "geography",
            },
            {
                "question": "Which element has the chemical symbol 'O'?",
                "choices": ["Gold", "Oxygen", "Osmium", "Olive"],
                "answer": 1,
                "domain": "science",
            },
            {
                "question": "What is 15 * 17?",
                "choices": ["245", "255", "265", "275"],
                "answer": 1,
                "domain": "math",
            },
            {
                "question": "Who wrote 'Romeo and Juliet'?",
                "choices": ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"],
                "answer": 1,
                "domain": "literature",
            },
            {
                "question": "What is the largest planet in our solar system?",
                "choices": ["Earth", "Mars", "Jupiter", "Saturn"],
                "answer": 2,
                "domain": "science",
            },
            {
                "question": "In which year did World War II end?",
                "choices": ["1943", "1944", "1945", "1946"],
                "answer": 2,
                "domain": "history",
            },
            {
                "question": "What is the square root of 144?",
                "choices": ["10", "11", "12", "13"],
                "answer": 2,
                "domain": "math",
            },
            {
                "question": "Which programming language is known as the 'language of the web'?",
                "choices": ["Python", "Java", "JavaScript", "C++"],
                "answer": 2,
                "domain": "technology",
            },
            {
                "question": "What is the freezing point of water in Celsius?",
                "choices": ["-10", "0", "10", "100"],
                "answer": 1,
                "domain": "science",
            },
            {
                "question": "Who painted the Mona Lisa?",
                "choices": ["Vincent van Gogh", "Pablo Picasso", "Leonardo da Vinci", "Michelangelo"],
                "answer": 2,
                "domain": "art",
            },
        ]
    
    def evaluate(self, verbose: bool = False) -> BenchmarkResult:
        """Evaluate language understanding"""
        correct = 0
        total = len(self.test_questions)
        domain_scores = {}
        
        details = []
        
        for q in tqdm(self.test_questions, desc="Language Understanding"):
            # Format question
            prompt = self._format_question(q)
            
            # Generate answer
            try:
                if hasattr(self.model, 'generate'):
                    response = self.model.generate(
                        prompt=prompt,
                        max_new_tokens=10,
                        temperature=0.1,
                    )
                else:
                    response = "A"  # Default fallback
                
                # Parse answer
                predicted = self._parse_answer(response, len(q["choices"]))
                is_correct = (predicted == q["answer"])
                
                if is_correct:
                    correct += 1
                
                # Track domain scores
                domain = q["domain"]
                if domain not in domain_scores:
                    domain_scores[domain] = {"correct": 0, "total": 0}
                domain_scores[domain]["total"] += 1
                if is_correct:
                    domain_scores[domain]["correct"] += 1
                
                details.append({
                    "question": q["question"],
                    "predicted": predicted,
                    "correct": q["answer"],
                    "is_correct": is_correct,
                    "domain": domain,
                })
                
            except Exception as e:
                if verbose:
                    print(f"Error on question: {e}")
                details.append({
                    "question": q["question"],
                    "error": str(e),
                })
        
        # Calculate domain percentages
        domain_percentages = {
            domain: (scores["correct"] / scores["total"] * 100)
            for domain, scores in domain_scores.items()
        }
        
        return BenchmarkResult(
            benchmark_name="Language Understanding",
            score=correct,
            max_score=total,
            metrics={
                "accuracy": correct / total * 100 if total > 0 else 0,
                "domain_scores": domain_percentages,
            },
            timestamp=datetime.now(),
            details=details,
        )
    
    def _format_question(self, q: Dict) -> str:
        """Format question for model"""
        prompt = f"Question: {q['question']}\n"
        for i, choice in enumerate(q["choices"]):
            prompt += f"{chr(65 + i)}. {choice}\n"
        prompt += "Answer:"
        return prompt
    
    def _parse_answer(self, response: str, num_choices: int) -> int:
        """Parse model response to get answer index"""
        response = response.strip().upper()
        
        # Look for letter answer
        for i in range(num_choices):
            if chr(65 + i) in response[:10]:  # Check first 10 chars
                return i
        
        # Look for number answer
        for i in range(num_choices):
            if str(i) in response[:10]:
                return i
        
        return 0  # Default to first choice


class ReasoningBenchmark:
    """
    Reasoning ability benchmark
    推理能力基准测试
    """
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        self.reasoning_problems = [
            {
                "problem": "If all roses are flowers and some flowers fade quickly, then:",
                "choices": [
                    "All roses fade quickly",
                    "Some roses fade quickly",
                    "No roses fade quickly",
                    "Cannot be determined",
                ],
                "answer": 3,
                "type": "logical",
            },
            {
                "problem": "John is taller than Mary. Mary is taller than Sue. Who is the shortest?",
                "choices": ["John", "Mary", "Sue", "Cannot be determined"],
                "answer": 2,
                "type": "transitive",
            },
            {
                "problem": "What comes next in the sequence: 2, 4, 8, 16, ...",
                "choices": ["24", "30", "32", "64"],
                "answer": 2,
                "type": "pattern",
            },
            {
                "problem": "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
                "choices": ["5 minutes", "100 minutes", "500 minutes", "20 minutes"],
                "answer": 0,
                "type": "mathematical",
            },
            {
                "problem": "A bat and a ball cost $11 in total. The bat costs $10 more than the ball. How much does the ball cost?",
                "choices": ["$0.50", "$1", "$1.50", "$2"],
                "answer": 0,
                "type": "mathematical",
            },
        ]
    
    def evaluate(self, verbose: bool = False) -> BenchmarkResult:
        """Evaluate reasoning ability"""
        correct = 0
        total = len(self.reasoning_problems)
        type_scores = {}
        
        details = []
        
        for problem in tqdm(self.reasoning_problems, desc="Reasoning"):
            try:
                # Format prompt
                prompt = f"Problem: {problem['problem']}\n"
                for i, choice in enumerate(problem["choices"]):
                    prompt += f"{chr(65 + i)}. {choice}\n"
                prompt += "Answer:"
                
                # Generate response
                if hasattr(self.model, 'generate'):
                    response = self.model.generate(
                        prompt=prompt,
                        max_new_tokens=20,
                        temperature=0.1,
                    )
                else:
                    response = "A"
                
                # Parse answer
                predicted = self._parse_answer(response, len(problem["choices"]))
                is_correct = (predicted == problem["answer"])
                
                if is_correct:
                    correct += 1
                
                # Track by type
                ptype = problem["type"]
                if ptype not in type_scores:
                    type_scores[ptype] = {"correct": 0, "total": 0}
                type_scores[ptype]["total"] += 1
                if is_correct:
                    type_scores[ptype]["correct"] += 1
                
                details.append({
                    "problem": problem["problem"][:50] + "...",
                    "predicted": predicted,
                    "correct": problem["answer"],
                    "is_correct": is_correct,
                    "type": ptype,
                })
                
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
                details.append({
                    "problem": problem["problem"][:50] + "...",
                    "error": str(e),
                })
        
        type_percentages = {
            t: (s["correct"] / s["total"] * 100)
            for t, s in type_scores.items()
        }
        
        return BenchmarkResult(
            benchmark_name="Reasoning",
            score=correct,
            max_score=total,
            metrics={
                "accuracy": correct / total * 100 if total > 0 else 0,
                "type_scores": type_percentages,
            },
            timestamp=datetime.now(),
            details=details,
        )
    
    def _parse_answer(self, response: str, num_choices: int) -> int:
        """Parse answer from response"""
        response = response.strip().upper()
        
        for i in range(num_choices):
            if chr(65 + i) in response[:20]:
                return i
        
        return 0


class MemoryBenchmark:
    """
    Memory system benchmark
    记忆系统基准测试
    """
    
    def __init__(self, memory_system):
        self.memory = memory_system
    
    def evaluate(self, verbose: bool = False) -> BenchmarkResult:
        """Evaluate memory performance"""
        import torch
        
        # Test 1: Storage capacity
        num_items = 100
        storage_times = []
        
        for i in range(num_items):
            content = f"Test memory item number {i} with some content to make it realistic."
            vector = torch.randn(768)
            
            start = time.time()
            if hasattr(self.memory, 'encode'):
                self.memory.encode(vector, content)
            elif hasattr(self.memory, 'add'):
                self.memory.add(content, vector.numpy())
            storage_times.append(time.time() - start)
        
        avg_storage_time = np.mean(storage_times) * 1000  # ms
        
        # Test 2: Retrieval speed
        query = torch.randn(768)
        retrieval_times = []
        
        for _ in range(10):
            start = time.time()
            if hasattr(self.memory, 'retrieve'):
                self.memory.retrieve(query, top_k=5)
            elif hasattr(self.memory, 'search'):
                self.memory.search(query.numpy(), k=5)
            retrieval_times.append(time.time() - start)
        
        avg_retrieval_time = np.mean(retrieval_times) * 1000  # ms
        
        # Test 3: Retrieval accuracy
        # Store specific items and try to retrieve them
        test_items = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language",
        ]
        
        accuracy_scores = []
        for item in test_items:
            vector = torch.randn(768)
            if hasattr(self.memory, 'encode'):
                self.memory.encode(vector, item)
            elif hasattr(self.memory, 'add'):
                self.memory.add(item, vector.numpy())
            
            # Retrieve
            if hasattr(self.memory, 'retrieve'):
                results = self.memory.retrieve(vector, top_k=5)
            elif hasattr(self.memory, 'search'):
                results = self.memory.search(vector.numpy(), k=5)
            else:
                results = []
            
            # Check if item is in results
            found = False
            for r in results:
                content = r.raw_content if hasattr(r, 'raw_content') else r.get('content', '')
                if item in str(content):
                    found = True
                    break
            
            accuracy_scores.append(1.0 if found else 0.0)
        
        retrieval_accuracy = np.mean(accuracy_scores) * 100
        
        return BenchmarkResult(
            benchmark_name="Memory System",
            score=retrieval_accuracy,
            max_score=100,
            metrics={
                "storage_time_ms": avg_storage_time,
                "retrieval_time_ms": avg_retrieval_time,
                "retrieval_accuracy": retrieval_accuracy,
                "items_stored": num_items,
            },
            timestamp=datetime.now(),
        )


class SpeedBenchmark:
    """
    Inference speed benchmark
    推理速度基准测试
    """
    
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, verbose: bool = False) -> BenchmarkResult:
        """Evaluate inference speed"""
        test_prompts = [
            "Hello, how are you?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about artificial intelligence.",
            "What are the benefits of exercise?",
            "Describe the process of photosynthesis.",
        ]
        
        tokens_per_second_list = []
        generation_times = []
        
        for prompt in tqdm(test_prompts, desc="Speed Test"):
            try:
                start = time.time()
                
                if hasattr(self.model, 'generate'):
                    response = self.model.generate(
                        prompt=prompt,
                        max_new_tokens=50,
                        temperature=0.7,
                    )
                else:
                    response = "Test response"
                
                elapsed = time.time() - start
                
                # Estimate tokens (rough approximation)
                num_tokens = len(response.split()) + len(prompt.split())
                tps = num_tokens / elapsed if elapsed > 0 else 0
                
                tokens_per_second_list.append(tps)
                generation_times.append(elapsed)
                
            except Exception as e:
                if verbose:
                    print(f"Error: {e}")
        
        avg_tps = np.mean(tokens_per_second_list) if tokens_per_second_list else 0
        avg_time = np.mean(generation_times) if generation_times else 0
        
        return BenchmarkResult(
            benchmark_name="Inference Speed",
            score=avg_tps,
            max_score=100,  # Target: 100 tokens/sec
            metrics={
                "tokens_per_second": avg_tps,
                "avg_generation_time": avg_time,
                "max_tokens_per_second": max(tokens_per_second_list) if tokens_per_second_list else 0,
                "min_tokens_per_second": min(tokens_per_second_list) if tokens_per_second_list else 0,
            },
            timestamp=datetime.now(),
        )


class ComprehensiveBenchmark:
    """
    Comprehensive benchmark suite
    综合基准测试套件
    """
    
    def __init__(
        self,
        model,
        tokenizer=None,
        memory_system=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.memory_system = memory_system
        
        self.benchmarks = []
        self.results: List[BenchmarkResult] = []
    
    def add_benchmark(self, benchmark):
        """Add a benchmark to the suite"""
        self.benchmarks.append(benchmark)
    
    def run_all(self, verbose: bool = False) -> Dict[str, Any]:
        """Run all benchmarks"""
        print("=" * 60)
        print("Running Comprehensive Benchmark Suite")
        print("=" * 60)
        
        # Language understanding
        if self.tokenizer:
            lang_benchmark = LanguageUnderstandingBenchmark(self.model, self.tokenizer)
            result = lang_benchmark.evaluate(verbose)
            self.results.append(result)
            print(f"\nLanguage Understanding: {result.score}/{result.max_score} "
                  f"({result.metrics['accuracy']:.2f}%)")
        
        # Reasoning
        if self.tokenizer:
            reasoning_benchmark = ReasoningBenchmark(self.model, self.tokenizer)
            result = reasoning_benchmark.evaluate(verbose)
            self.results.append(result)
            print(f"Reasoning: {result.score}/{result.max_score} "
                  f"({result.metrics['accuracy']:.2f}%)")
        
        # Memory
        if self.memory_system:
            memory_benchmark = MemoryBenchmark(self.memory_system)
            result = memory_benchmark.evaluate(verbose)
            self.results.append(result)
            print(f"Memory: {result.score:.2f}% accuracy, "
                  f"{result.metrics['retrieval_time_ms']:.2f}ms retrieval")
        
        # Speed
        speed_benchmark = SpeedBenchmark(self.model)
        result = speed_benchmark.evaluate(verbose)
        self.results.append(result)
        print(f"Speed: {result.metrics['tokens_per_second']:.2f} tokens/sec")
        
        return self.get_summary()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get benchmark summary"""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "overall_score": 0,
        }
        
        # Calculate overall score
        if self.results:
            total_percentage = sum(
                r.score / r.max_score * 100 if r.max_score > 0 else 0
                for r in self.results
            )
            summary["overall_score"] = total_percentage / len(self.results)
        
        return summary
    
    def save_results(self, path: str):
        """Save results to file"""
        summary = self.get_summary()
        
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to: {path}")
    
    def compare_with(self, other_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with another benchmark result"""
        comparison = {
            "current": self.get_summary(),
            "baseline": other_results,
            "improvements": {},
        }
        
        # Compare individual benchmarks
        current_results = {r.benchmark_name: r for r in self.results}
        
        for baseline_result in other_results.get("results", []):
            name = baseline_result["benchmark_name"]
            if name in current_results:
                current = current_results[name]
                baseline_score = baseline_result["score"] / baseline_result["max_score"] * 100
                current_score = current.score / current.max_score * 100 if current.max_score > 0 else 0
                
                comparison["improvements"][name] = {
                    "baseline": baseline_score,
                    "current": current_score,
                    "improvement": current_score - baseline_score,
                }
        
        return comparison


def run_benchmark(
    model,
    tokenizer=None,
    memory_system=None,
    output_path: str = "benchmark_results.json",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark
    
    Args:
        model: Model to benchmark
        tokenizer: Tokenizer (optional)
        memory_system: Memory system (optional)
        output_path: Path to save results
        verbose: Verbose output
        
    Returns:
        Benchmark summary
    """
    benchmark = ComprehensiveBenchmark(model, tokenizer, memory_system)
    summary = benchmark.run_all(verbose)
    benchmark.save_results(output_path)
    
    print("\n" + "=" * 60)
    print(f"Overall Score: {summary['overall_score']:.2f}%")
    print("=" * 60)
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("Benchmark module - import and use with your model")
