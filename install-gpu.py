#!/usr/bin/env python3
"""
GPU Installation Script for nice-tts

This script installs GPU-enabled PyTorch dependencies for better performance.
Run this after setting up the virtual environment.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return True if successful"""
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {cmd}")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("🚀 Installing GPU-enabled PyTorch for nice-tts")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in a virtual environment")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Install GPU-enabled PyTorch
    print("\n📦 Installing PyTorch with CUDA 12.6 support...")
    pytorch_cmd = "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126"
    
    if not run_command(pytorch_cmd):
        print("❌ Failed to install PyTorch with GPU support")
        return False
    
    # Install other dependencies
    print("\n📦 Installing other dependencies...")
    if not run_command("uv sync"):
        print("❌ Failed to install other dependencies")
        return False
    
    # Verify GPU support
    print("\n🔍 Verifying GPU support...")
    verify_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA GPUs detected")
"""
    
    try:
        result = subprocess.run([sys.executable, "-c", verify_script], 
                              capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("❌ Could not verify PyTorch installation")
        return False
    
    print("\n✅ GPU installation completed successfully!")
    print("\n🎯 Next steps:")
    print("1. Copy .env.example to .env and configure your API keys")
    print("2. Run 'nice-tts check-gpu' to verify GPU acceleration")
    print("3. Start processing audio files with 'nice-tts process <audio_file>'")
    
    return True

if __name__ == "__main__":
    main()