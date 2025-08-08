#!/usr/bin/env python3
"""
Setup script for ICICLE-Benchmark V2
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Setup ICICLE-Benchmark V2')
    parser.add_argument('--install-deps', action='store_true', 
                       help='Install Python dependencies')
    parser.add_argument('--create-dirs', action='store_true',
                       help='Create necessary directories')
    parser.add_argument('--check-env', action='store_true',
                       help='Check environment and dependencies')
    parser.add_argument('--all', action='store_true',
                       help='Run all setup steps')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    print("🔧 ICICLE-Benchmark V2 Setup")
    print("=" * 50)
    
    if args.all or args.create_dirs:
        create_directories()
    
    if args.all or args.install_deps:
        install_dependencies()
    
    if args.all or args.check_env:
        check_environment()
    
    print("\n✅ Setup completed!")
    print("\nNext steps:")
    print("1. Configure your camera dataset in configs/cameras/")
    print("2. Run: python main.py --camera YOUR_CAMERA --config configs/experiments/baseline.yaml")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        'logs',
        'logs/debug',
        'data',
        'pretrained_weights',
        'outputs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], check=True)
        print("  ✓ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("  ❌ Failed to install dependencies")
        print("  Please install manually: pip install -r requirements.txt")


def check_environment():
    """Check environment and key dependencies."""
    print("\n🔍 Checking environment...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version >= (3, 8):
        print(f"  ✓ Python {python_version.major}.{python_version.minor}")
    else:
        print(f"  ❌ Python {python_version.major}.{python_version.minor} (requires 3.8+)")
    
    # Check key imports
    imports_to_check = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('sklearn', 'scikit-learn'),
    ]
    
    for module_name, display_name in imports_to_check:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} (not installed)")
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available (GPUs: {torch.cuda.device_count()})")
        else:
            print("  ⚠️  CUDA not available (CPU only)")
    except ImportError:
        print("  ❌ Cannot check CUDA (PyTorch not installed)")


if __name__ == '__main__':
    main()
