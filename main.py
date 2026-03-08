#!/usr/bin/env python3
"""
Brain-Inspired AI - Main Entry Point
类脑人工智能 - 主入口

Usage:
    python main.py [command] [options]

Commands:
    run         Run interactive mode
    server      Start API server
    train       Train the model
    evaluate    Run benchmarks
    web         Launch web interface
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def interactive_mode(args):
    """Run interactive mode"""
    print("🧠 Brain-Inspired AI - Interactive Mode")
    print("=" * 60)
    
    try:
        from src.models.model_integration import BrainInspiredModel
        from src.inference.streaming_inference import InferencePipeline
        
        print("Loading model...")
        model = BrainInspiredModel(
            base_model_name=args.model or "Qwen/Qwen3.5-0.8B",
            device=args.device or "auto",
            use_snn=not args.no_snn,
            use_memory=not args.no_memory,
        )
        
        pipeline = InferencePipeline(model)
        
        print("\nModel loaded! Type 'exit' to quit, 'clear' to clear history.\n")
        
        session_id = "interactive"
        pipeline.create_session(session_id)
        
        while True:
            try:
                user_input = input("👤 You: ").strip()
                
                if user_input.lower() == 'exit':
                    break
                
                if user_input.lower() == 'clear':
                    pipeline.sessions[session_id].clear_history()
                    print("History cleared.\n")
                    continue
                
                if not user_input:
                    continue
                
                print("🤖 AI: ", end="", flush=True)
                
                response = []
                for token in pipeline.generate(user_input, session_id=session_id):
                    if not token.is_special:
                        print(token.token, end="", flush=True)
                        response.append(token.token)
                
                print("\n")
                
            except KeyboardInterrupt:
                print("\nUse 'exit' to quit.")
                continue
            except Exception as e:
                print(f"\nError: {e}\n")
                continue
        
        print("\nGoodbye!")
        
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


def server_mode(args):
    """Start API server"""
    print("🚀 Starting API Server...")
    
    try:
        from src.api.fastapi_server import start_server
        start_server(
            host=args.host or "0.0.0.0",
            port=args.port or 8000,
            reload=args.reload,
        )
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install FastAPI: pip install fastapi uvicorn")
        sys.exit(1)


def train_mode(args):
    """Train the model"""
    print("🎓 Training Mode")
    print("=" * 60)
    
    try:
        from src.models.model_integration import BrainInspiredModel
        from src.training.offline_training import (
            OfflineTrainer,
            TrainingConfig,
            ModuleTrainer,
        )
        
        # Load model
        print("Loading model...")
        model = BrainInspiredModel(
            base_model_name=args.model or "Qwen/Qwen3.5-0.8B",
            device=args.device or "auto",
        )
        
        # Training configuration
        config = TrainingConfig(
            epochs=args.epochs or 3,
            batch_size=args.batch_size or 32,
            learning_rate=args.learning_rate or 5e-5,
            output_dir=args.output_dir or "./checkpoints",
        )
        
        if args.module:
            # Train specific module
            print(f"Training module: {args.module}")
            trainer = ModuleTrainer(device=model.device)
            
            if args.module == "snn":
                # Generate dummy spike data for demonstration
                import torch
                spike_data = [torch.randn(10, 768) for _ in range(100)]
                result = trainer.train_snn(model.snn_input, spike_data)
                print(f"SNN training complete: {result}")
                
            elif args.module == "memory":
                import torch
                memory_data = [
                    (torch.randn(768), torch.randn(768))
                    for _ in range(100)
                ]
                result = trainer.train_memory(model.hippocampus, memory_data)
                print(f"Memory training complete: {result}")
        else:
            # Full training
            print("Starting full training...")
            # This would require a dataset
            print("Note: Full training requires a dataset. Use --data to specify.")
        
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def evaluate_mode(args):
    """Run benchmarks"""
    print("📊 Running Benchmarks")
    print("=" * 60)
    
    try:
        from src.models.model_integration import BrainInspiredModel
        from tests.benchmark import run_benchmark
        
        print("Loading model...")
        model = BrainInspiredModel(
            base_model_name=args.model or "Qwen/Qwen3.5-0.8B",
            device=args.device or "auto",
        )
        
        print("Running benchmarks...")
        results = run_benchmark(
            model=model,
            tokenizer=model.tokenizer,
            memory_system=model.hippocampus if args.test_memory else None,
            output_path=args.output or "benchmark_results.json",
            verbose=args.verbose,
        )
        
        print("\nBenchmark complete!")
        
    except ImportError as e:
        print(f"Error: {e}")
        sys.exit(1)


def web_mode(args):
    """Launch web interface"""
    print("🌐 Launching Web Interface...")
    
    try:
        import streamlit.web.bootstrap as bootstrap
        import os
        
        script_path = os.path.join(
            os.path.dirname(__file__),
            "frontend",
            "streamlit_app.py",
        )
        
        sys.argv = ["streamlit", "run", script_path]
        bootstrap.run(
            script_path,
            "Brain-Inspired AI",
            [],
            {},
        )
        
    except ImportError:
        print("Error: Streamlit not installed.")
        print("Install with: pip install streamlit")
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Brain-Inspired AI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py run                    # Interactive mode
    python main.py server --port 8000     # Start API server
    python main.py train --epochs 5       # Train model
    python main.py evaluate               # Run benchmarks
    python main.py web                    # Launch web interface
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run interactive mode")
    run_parser.add_argument("--model", type=str, help="Model name")
    run_parser.add_argument("--device", type=str, help="Device (cpu/cuda)")
    run_parser.add_argument("--no-snn", action="store_true", help="Disable SNN")
    run_parser.add_argument("--no-memory", action="store_true", help="Disable memory")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start API server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=8000)
    server_parser.add_argument("--reload", action="store_true")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--model", type=str, help="Model name")
    train_parser.add_argument("--device", type=str, help="Device")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--output-dir", type=str, help="Output directory")
    train_parser.add_argument("--module", type=str, help="Train specific module (snn/memory)")
    train_parser.add_argument("--data", type=str, help="Training data path")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmarks")
    eval_parser.add_argument("--model", type=str, help="Model name")
    eval_parser.add_argument("--device", type=str, help="Device")
    eval_parser.add_argument("--output", type=str, help="Output file")
    eval_parser.add_argument("--test-memory", action="store_true", help="Test memory system")
    eval_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # Web command
    web_parser = subparsers.add_parser("web", help="Launch web interface")
    web_parser.add_argument("--port", type=int, help="Port")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    commands = {
        "run": interactive_mode,
        "server": server_mode,
        "train": train_mode,
        "evaluate": evaluate_mode,
        "web": web_mode,
    }
    
    if args.command in commands:
        commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
