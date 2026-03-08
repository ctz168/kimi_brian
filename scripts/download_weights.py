#!/usr/bin/env python3
"""
Download pre-trained weights script
下载预训练权重脚本
"""

import os
import sys
import argparse
from pathlib import Path
import torch
from huggingface_hub import snapshot_download, hf_hub_download


def download_base_model(model_name: str = "Qwen/Qwen3.5-0.8B", cache_dir: str = "./weights"):
    """Download base language model"""
    print(f"Downloading base model: {model_name}")
    
    try:
        snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, "base_model"),
            local_dir_use_symlinks=False,
        )
        print(f"✓ Base model downloaded to {cache_dir}/base_model")
        return True
    except Exception as e:
        print(f"✗ Error downloading base model: {e}")
        return False


def download_vision_model(cache_dir: str = "./weights"):
    """Download vision model (CLIP)"""
    print("Downloading vision model: openai/clip-vit-base-patch32")
    
    try:
        snapshot_download(
            repo_id="openai/clip-vit-base-patch32",
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, "vision_model"),
            local_dir_use_symlinks=False,
        )
        print(f"✓ Vision model downloaded to {cache_dir}/vision_model")
        return True
    except Exception as e:
        print(f"✗ Error downloading vision model: {e}")
        return False


def download_brain_components(cache_dir: str = "./weights"):
    """Download brain-inspired components (if available on HuggingFace)"""
    print("Checking for brain components...")
    
    # These would be custom trained components
    # For now, they will be initialized randomly
    print("ℹ Brain components will be initialized during first run")
    return True


def verify_weights(cache_dir: str = "./weights") -> dict:
    """Verify downloaded weights"""
    status = {
        "base_model": False,
        "vision_model": False,
        "brain_components": False,
    }
    
    # Check base model
    base_path = os.path.join(cache_dir, "base_model")
    if os.path.exists(base_path):
        required_files = ["config.json", "pytorch_model.bin"]
        status["base_model"] = all(
            os.path.exists(os.path.join(base_path, f))
            for f in required_files
        )
    
    # Check vision model
    vision_path = os.path.join(cache_dir, "vision_model")
    if os.path.exists(vision_path):
        status["vision_model"] = os.path.exists(
            os.path.join(vision_path, "config.json")
        )
    
    return status


def main():
    parser = argparse.ArgumentParser(description="Download pre-trained weights")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./weights",
        help="Directory to store weights",
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Download only base model",
    )
    parser.add_argument(
        "--vision-only",
        action="store_true",
        help="Download only vision model",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing weights",
    )
    
    args = parser.parse_args()
    
    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    
    if args.verify:
        print("Verifying existing weights...")
        status = verify_weights(args.cache_dir)
        for component, exists in status.items():
            symbol = "✓" if exists else "✗"
            print(f"{symbol} {component}")
        return
    
    print("=" * 60)
    print("Downloading Pre-trained Weights")
    print("=" * 60)
    
    results = []
    
    # Download base model
    if not args.vision_only:
        results.append(("Base Model", download_base_model(cache_dir=args.cache_dir)))
    
    # Download vision model
    if not args.base_only:
        results.append(("Vision Model", download_vision_model(cache_dir=args.cache_dir)))
    
    # Download brain components
    if not args.base_only and not args.vision_only:
        results.append(("Brain Components", download_brain_components(cache_dir=args.cache_dir)))
    
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)
    
    for name, success in results:
        symbol = "✓" if success else "✗"
        print(f"{symbol} {name}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✓ All weights downloaded successfully!")
        print(f"Weights stored in: {os.path.abspath(args.cache_dir)}")
    else:
        print("\n✗ Some downloads failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
