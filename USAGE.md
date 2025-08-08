# ICICLE-Benchmark V2 - Usage Guide

## Quick Start

### 1. Setup
```bash
# Install dependencies and setup directories
python scripts/setup.py --all

# Or step by step:
python scripts/setup.py --install-deps
python scripts/setup.py --create-dirs
python scripts/setup.py --check-env
```

### 2. Basic Usage

#### Using Config File (Recommended)
```bash
# Run with baseline configuration
python main.py --camera APN_K024 --config configs/experiments/baseline.yaml

# Run with advanced configuration
python main.py --camera APN_K024 --config configs/experiments/advanced.yaml
```

#### Using Command Line Arguments
```bash
# Simple run with minimal arguments
python main.py --camera APN_K024 --model bioclip --epochs 30

# More detailed configuration
python main.py \
    --camera APN_K024 \
    --model bioclip \
    --epochs 30 \
    --lr 0.0001 \
    --batch_size 128 \
    --ood_method uncertainty \
    --al_method active_ft \
    --cl_method ewc \
    --use_peft \
    --calibration \
    --gpu_cleanup
```

## Configuration System

### Priority Order
1. Command-line arguments (highest priority)
2. Experiment config file (if provided)
3. Camera-specific config file
4. Default values (lowest priority)

### Camera Configuration
Create `configs/cameras/YOUR_CAMERA.yaml`:
```yaml
camera: YOUR_CAMERA
class_names:
  - class1
  - class2
  - class3
train_data_path: "path/to/train.json"
eval_data_path: "path/to/eval.json"
```

### Experiment Configuration
Create custom experiment configs in `configs/experiments/`:
```yaml
experiment_name: "my_experiment"
model: "bioclip"
epochs: 50
learning_rate: 0.0001

# Module configurations
ood_method: "uncertainty"
ood_params:
  uncertainty_threshold: 0.7

al_method: "active_ft"
al_params:
  selection_ratio: 0.3

cl_method: "ewc" 
cl_params:
  ewc_lambda: 0.4

# Enable advanced features
use_peft: true
calibration: true
gpu_cleanup: true
```

## Core Modules

### 1. Out-of-Distribution (OOD) Detection
- **`none`**: No OOD detection
- **`all`**: All samples considered OOD 
- **`oracle`**: Use true labels to identify OOD
- **`uncertainty`**: Entropy/confidence based detection

### 2. Active Learning (AL)
- **`none`**: No sample selection
- **`all`**: Select all available samples
- **`random`**: Random sample selection
- **`uncertainty`**: Select most uncertain samples
- **`active_ft`**: Feature-based diverse sampling
- **`mls`**: Minimum Logit Score
- **`msp`**: Minimum Softmax Probability

### 3. Continual Learning (CL)
- **`none`**: No training (zero-shot)
- **`naive-ft`**: Naive fine-tuning on new samples
- **`accumulative`**: Train on all samples seen so far
- **`ewc`**: Elastic Weight Consolidation
- **`replay`**: Experience replay buffer

### 4. Parameter-Efficient Fine-Tuning (PEFT)
- **LoRA**: Low-Rank Adaptation
- Enabled with `--use_peft` flag

### 5. Calibration
- **Temperature Scaling**: Post-hoc calibration
- **Platt Scaling**: Sigmoid-based calibration  
- **Isotonic Regression**: Non-parametric calibration
- Enabled with `--calibration` flag

## Advanced Features

### GPU Memory Management
```bash
# Enable automatic GPU VRAM cleaning
python main.py --camera APN_K024 --config baseline.yaml --gpu_cleanup
```

### Debug Mode
```bash
# Enable verbose logging and debug output
python main.py --camera APN_K024 --config baseline.yaml --debug
```

### Evaluation Only
```bash
# Skip training, only run evaluation
python main.py --camera APN_K024 --config baseline.yaml --eval_only
```

## Output Structure

Results are saved to the specified output directory:
```
logs/
├── CAMERA_NAME/
│   ├── EXPERIMENT_NAME/
│   │   ├── config.yaml              # Final merged configuration
│   │   ├── EXPERIMENT_NAME.log      # Training logs
│   │   ├── final_results.pkl        # Complete results
│   │   ├── model_ckp_0.pth         # Model checkpoints
│   │   ├── results_ckp_0.pkl       # Per-checkpoint results
│   │   └── ...
│   └── debug_TIMESTAMP/             # Debug runs
└── ...
```

## Examples

### Run Examples
```bash
# Baseline example
python scripts/example.py baseline

# Advanced example with all features
python scripts/example.py advanced

# CLI-based example
python scripts/example.py cli
```

### Custom Camera Setup
1. Create camera config: `configs/cameras/my_camera.yaml`
2. Specify data paths and class names
3. Run: `python main.py --camera my_camera --config baseline.yaml`

### Custom Experiment
1. Create experiment config: `configs/experiments/my_exp.yaml`
2. Configure methods and hyperparameters
3. Run: `python main.py --camera APN_K024 --config my_exp.yaml`

## Troubleshooting

### Common Issues
1. **Import errors**: Run `python scripts/setup.py --install-deps`
2. **CUDA errors**: Check GPU availability with `python scripts/setup.py --check-env`
3. **Config errors**: Verify YAML syntax and file paths
4. **Memory issues**: Enable `--gpu_cleanup` flag

### Debug Mode
Enable debug mode for detailed logging:
```bash
python main.py --camera APN_K024 --config baseline.yaml --debug
```

### Minimal Test Run
For quick testing with reduced parameters:
```bash
python main.py --camera APN_K024 --epochs 2 --debug --no_save
```
