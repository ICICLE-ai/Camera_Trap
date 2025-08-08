#!/usr/bin/env python3
"""
Example usage script for ICICLE-Benchmark V2
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_baseline_example():
    """Run a baseline example."""
    
    print("🎯 Running ICICLE-Benchmark V2 Baseline Example")
    print("=" * 60)
    
    # Import main components
    try:
        from main import main
        import argparse
        
        # Mock command line arguments
        sys.argv = [
            'example.py',
            '--camera', 'APN_K024',
            '--config', 'configs/experiments/baseline.yaml',
            '--debug'
        ]
        
        print("Configuration:")
        print(f"  Camera: APN_K024")
        print(f"  Config: configs/experiments/baseline.yaml")
        print(f"  Debug: enabled")
        print()
        
        # Run the pipeline
        results = main()
        
        print("\n✅ Example completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        print("\nThis is expected if dependencies are not installed.")
        print("Run: python scripts/setup.py --all")
        return None


def run_advanced_example():
    """Run an advanced example with all features."""
    
    print("🚀 Running ICICLE-Benchmark V2 Advanced Example")
    print("=" * 60)
    
    try:
        from main import main
        import argparse
        
        # Mock command line arguments for advanced config
        sys.argv = [
            'example.py',
            '--camera', 'APN_K024',
            '--config', 'configs/experiments/advanced.yaml',
            '--debug'
        ]
        
        print("Configuration:")
        print(f"  Camera: APN_K024")
        print(f"  Config: configs/experiments/advanced.yaml")
        print(f"  Features: OOD, AL, CL, PEFT, Calibration")
        print(f"  Debug: enabled")
        print()
        
        # Run the pipeline
        results = main()
        
        print("\n✅ Advanced example completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ Advanced example failed: {str(e)}")
        print("\nThis is expected if dependencies are not installed.")
        print("Run: python scripts/setup.py --all")
        return None


def run_cli_example():
    """Run example using command-line arguments instead of config file."""
    
    print("⌨️  Running ICICLE-Benchmark V2 CLI Example")
    print("=" * 60)
    
    try:
        from main import main
        
        # Mock command line arguments without config file
        sys.argv = [
            'example.py',
            '--camera', 'APN_K024',
            '--model', 'bioclip',
            '--epochs', '5',  # Shorter for example
            '--lr', '0.0001',
            '--batch_size', '64',
            '--ood_method', 'uncertainty',
            '--al_method', 'random',
            '--cl_method', 'naive-ft',
            '--debug'
        ]
        
        print("Configuration (CLI args):")
        print(f"  Camera: APN_K024")
        print(f"  Model: bioclip")
        print(f"  Epochs: 5 (short for demo)")
        print(f"  Learning rate: 0.0001")
        print(f"  Methods: uncertainty OOD, random AL, naive-ft CL")
        print()
        
        # Run the pipeline
        results = main()
        
        print("\n✅ CLI example completed successfully!")
        return results
        
    except Exception as e:
        print(f"\n❌ CLI example failed: {str(e)}")
        print("\nThis is expected if dependencies are not installed.")
        return None


def main():
    """Main example runner."""
    
    if len(sys.argv) > 1:
        example_type = sys.argv[1].lower()
    else:
        example_type = 'baseline'
    
    if example_type == 'baseline':
        return run_baseline_example()
    elif example_type == 'advanced':
        return run_advanced_example()
    elif example_type == 'cli':
        return run_cli_example()
    else:
        print("Usage: python scripts/example.py [baseline|advanced|cli]")
        return None


if __name__ == '__main__':
    main()
