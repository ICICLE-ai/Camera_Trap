# ICICLE-Benchmark V2

A clean, organized, and scalable version of the ICICLE benchmark for camera trap image analysis with adaptive learning capabilities.

## Features

- **Clean Architecture**: Modular design with clear separation of concerns
- **Flexible Configuration**: Support for both YAML config files and command-line arguments
- **Core Modules**: OOD detection, Active Learning, Continual Learning, PEFT (LoRA), and Calibration
- **Efficient GPU Management**: Optional GPU VRAM cleaning with simple on/off toggle
- **Comprehensive Logging**: Clean and organized logging system
- **Scalable Design**: Easy to extend with new methods and datasets

## Quick Start

### Using Config File (Recommended)
```bash
python main.py --camera APN_K024 --config configs/experiments/baseline.yaml
```

### Using Command Line Arguments
```bash
python main.py --camera APN_K024 --model bioclip --epochs 30 --lr 0.0001
```

## Project Structure

```
├── main.py                    # Main entry point
├── configs/                   # Configuration files
│   ├── cameras/              # Camera-specific configs
│   └── experiments/          # Experiment configs
├── src/                      # Source code
│   ├── core/                 # Core components
│   ├── modules/              # AL, CL, OOD, PEFT modules
│   ├── models/               # Model implementations
│   ├── data/                 # Data handling
│   └── utils/                # Utilities
├── scripts/                  # Helper scripts
└── logs/                     # Log outputs
```

## Configuration

The system supports flexible configuration through YAML files or command-line arguments. When both are provided, config file takes precedence for overlapping parameters.

## Core Modules

1. **OOD Detection**: Out-of-distribution sample detection
2. **Active Learning**: Intelligent sample selection strategies
3. **Continual Learning**: Incremental learning approaches
4. **PEFT (LoRA)**: Parameter-efficient fine-tuning
5. **Calibration**: Inference-level model calibration
