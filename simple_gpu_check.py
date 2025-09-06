#!/usr/bin/env python3
"""Direct GPU check without CLI"""

import sys
from pathlib import Path

# Add src to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

try:
    import torch
    print(f"🔍 Checking GPU availability...\n")
    
    # PyTorch info
    print("PyTorch Information:")
    print(f"  Version: {torch.__version__}")
    
    # CUDA check
    if torch.cuda.is_available():
        print("\n✅ CUDA GPU Available")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Device Count: {torch.cuda.device_count()}")
        
        # Device details
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            
            print(f"\n  Device {i}: {props.name}")
            print(f"    Total Memory: {memory_total:.2f} GB")
            print(f"    Allocated: {memory_allocated:.2f} GB")
            print(f"    Reserved: {memory_reserved:.2f} GB")
        
        print("\n💡 GPU acceleration is available for transcription")
    else:
        print("\n⚠️  No CUDA GPU Available")
        print("  Transcription will use CPU (slower but still functional)")
        print("\n💡 To enable GPU acceleration:")
        print("  1. Install CUDA-compatible PyTorch")
        print("  2. Ensure NVIDIA drivers are installed")
        print("  3. Restart the application")

except ImportError as e:
    print(f"❌ Error importing torch: {e}")
except Exception as e:
    print(f"❌ Error checking GPU: {e}")