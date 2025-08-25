"""
Enhanced logging utilities for ICICLE-Benchmark V2
"""

import logging
import os
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, Any, Optional, List
import re
import unicodedata


class ICICLEColorFormatter(logging.Formatter):
    """Enhanced color formatter matching ICICLE-Benchmark reference style."""
    
    # ANSI color codes - matching ICICLE styles
    COLORS = {
        'DEBUG': '\033[38;5;245m',    # Gray
        'INFO': '\033[97m',           # Bright white
        'WARNING': '\033[93m',        # Yellow
        'ERROR': '\033[91m',          # Red
        'CRITICAL': '\033[95m',       # Magenta
        'RESET': '\033[0m'            # Reset
    }
    
    # Box drawing characters for structured display
    BOX_CHARS = {
        'top_left': '┌',
        'top_right': '┐',
        'bottom_left': '└',
        'bottom_right': '┘',
        'horizontal': '─',
        'vertical': '│',
        'cross': '├',
        'right_cross': '┤'
    }
    
    def __init__(self, use_colors: bool = True, use_timestamps: bool = False):
        super().__init__()
        self.use_colors = use_colors
        self.use_timestamps = use_timestamps
    
    def format(self, record):
        if self.use_timestamps:
            # Format with timestamp: 2025-08-14 16:38:31 - __main__ - INFO - message
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            prefix = f"{timestamp} - {record.name} - "
        else:
            # Format without timestamp: message only
            prefix = ""
        
        if self.use_colors:
            level_color = self.COLORS.get(record.levelname, self.COLORS['INFO'])
            level_formatted = f"{level_color}{record.levelname}\033[0m"
        else:
            level_formatted = record.levelname
        
        if self.use_timestamps:
            return f"{prefix}{level_formatted} - {record.getMessage()}"
        else:
            return record.getMessage()


class ICICLELogger:
    """Enhanced logger with structured formatting and organized sections."""
    
    def __init__(self, name: str = __name__, use_timestamps: bool = False):
        self.logger = logging.getLogger(name)
        self.use_timestamps = use_timestamps
        self.formatter = ICICLEColorFormatter(use_timestamps=use_timestamps)
        
    def setup_enhanced_logging(self, use_timestamps: bool = False):
        """Setup enhanced logging with colors and formatting for console only."""
        self.use_timestamps = use_timestamps
        self.formatter = ICICLEColorFormatter(use_timestamps=use_timestamps)
        
        # Create root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(self.formatter)
        root_logger.addHandler(console_handler)

    def setup_logging(self, output_dir: str, debug: bool = False, experiment_name: str = "experiment", use_timestamps: bool = False) -> logging.Logger:
        """Setup logging for the pipeline with ICICLE-style formatting."""
        self.use_timestamps = use_timestamps
        self.formatter = ICICLEColorFormatter(use_timestamps=use_timestamps)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging level
        level = logging.DEBUG if debug else logging.INFO
        
        # Get root logger
        logger = logging.getLogger()
        logger.setLevel(level)
        
        # Remove existing handlers to avoid duplicates
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler with color formatter
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(self.formatter)
        logger.addHandler(console_handler)
        
        # File handler with regular formatter (always with timestamps for files)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        log_file = os.path.join(output_dir, f"{experiment_name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Suppress verbose loggers for cleaner output
        self.suppress_verbose_loggers()
        
        # Silent logging initialization
        return logger

    def suppress_verbose_loggers(self):
        """Suppress verbose logging from external libraries."""
        # Suppress model loading details
        logging.getLogger('src.models.factory').setLevel(logging.WARNING)
        logging.getLogger('src.models.bioclip_model').setLevel(logging.WARNING)
        
        # Suppress open_clip verbose messages
        logging.getLogger('open_clip').setLevel(logging.WARNING)
        logging.getLogger('timm').setLevel(logging.WARNING)
        
        # Keep only essential messages from these modules
        essential_loggers = ['__main__', 'root']
        for logger_name in essential_loggers:
            logging.getLogger(logger_name).setLevel(logging.INFO)

    # Section Headers
    def log_section_header(self, title: str, style: str = 'major'):
        """Log structured section headers with consistent formatting."""
        logger = logging.getLogger()
        
        if style == 'major':
            # Major section with full width separator - clean and simple
            logger.info("")
            logger.info(f"\033[1;93m{'-' * 17} {title.upper()} {'-' * 17}\033[0m")
            logger.info("")
        elif style == 'phase':
            # Phase separator for pipeline steps
            logger.info("")
            logger.info(f"\033[38;5;16m{'=' * 80}\033[0m")
            logger.info(f"\033[38;5;16m{'=' * ((80 - len(title) - 2) // 2)} {title.upper()} {'=' * ((80 - len(title) - 2) // 2)}\033[0m")
            logger.info(f"\033[38;5;16m{'=' * 80}\033[0m")
            logger.info("")
        elif style == 'step':
            # Step headers for detailed processes - clean without emoji
            logger.info(f"\033[1;94mStep: {title}\033[0m")
        elif style == 'checkpoint':
            # Checkpoint evaluation progress
            logger.info(f"\033[38;5;236m{title}\033[0m")
        logger.info("")

    def log_info_box(self, title: str, content_dict: Dict[str, Any], box_style: str = 'cyan'):
        """Create structured information boxes from key-value dict (deprecated in favor of log_box)."""
        # Convert dict to simple "Key: Value" lines and delegate
        lines = [f"{k}: {v}" for k, v in content_dict.items()]
        self.log_box(title, lines, box_style)

    def log_box(self, title: str, lines: List[str], box_style: str = 'cyan'):
        """Render a boxed section with a title and a list of lines.
        Example:
        ┌──────┐
        │ Title│
        ├──────┤
        │ line │
        └──────┘
        """
        logger = logging.getLogger()
        colors = {
            'cyan': '\033[96m',
            'blue': '\033[94m', 
            'green': '\033[92m',
            'yellow': '\033[93m',
            'white': '\033[97m'
        }
        color = colors.get(box_style, colors['cyan'])
        reset = '\033[0m'

        # Normalize input and align colon-separated globally across all sections
        safe_lines = [str(s) for s in (lines or [])]
        safe_lines = self._align_colon_global(safe_lines)

        # Compute visual widths (emoji/unicode aware)
        content_visual = max([self._visual_len(title)] + [self._visual_len(s) for s in safe_lines])
        region_width = max(38, content_visual)  # region between the two internal spaces
        horiz_len = region_width + 2            # includes the two internal spaces

        horiz = '─' * horiz_len
        # Top
        logger.info("")
        logger.info(f"{color}┌{horiz}┐{reset}")
        # Title
        title_padded = self._pad_visual(title, region_width)
        logger.info(f"{color}│ {title_padded} │{reset}")
        # Divider
        logger.info(f"{color}├{horiz}┤{reset}")
        # Lines
        for s in safe_lines:
            # Support internal horizontal rules: '<hr>' or any line of only '─'
            if s.strip().lower() == '<hr>' or re.fullmatch(r"\s*─+\s*", s or ''):
                logger.info(f"{color}│ {'─' * region_width} │{reset}")
                continue
            s_padded = self._pad_visual(s, region_width)
            logger.info(f"{color}│ {s_padded} │{reset}")
        # Bottom
        logger.info(f"{color}└{horiz}┘{reset}")
        logger.info("")

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _strip_ansi(self, s: str) -> str:
        return self.ANSI_RE.sub('', s)

    def _visual_len(self, s: str) -> int:
        """Approximate display width: handles ANSI, emojis, CJK, and combining marks."""
        s = self._strip_ansi(s or '')
        width = 0
        for ch in s:
            # Skip zero-width combining marks
            if unicodedata.combining(ch):
                continue
            eaw = unicodedata.east_asian_width(ch)
            if eaw in ('W', 'F'):
                width += 2
            else:
                # Many emojis are 'Na' but display wide; heuristic: treat non-ASCII symbols as width 2
                width += 2 if ord(ch) > 0xFF and unicodedata.category(ch).startswith('So') else 1
        return width

    def _pad_visual(self, s: str, target: int) -> str:
        """Pad the string with spaces so its visual length equals target."""
        s = s or ''
        current = self._visual_len(s)
        if current >= target:
            return s
        return s + (' ' * (target - current))

    def _align_colon_blocks(self, lines: List[str]) -> List[str]:
        """Align values after ':' so they start at the same column within each block.
        Blocks are contiguous groups separated by blank lines.
        """
        out: List[str] = []
        block: List[str] = []

        def flush_block():
            if not block:
                return
            # Extract key/value pairs
            parts = []
            max_key = 0
            for ln in block:
                m = re.match(r"^(.*?):\s*(.*)$", ln)
                if m:
                    key = m.group(1)
                    val = m.group(2)
                    parts.append((key, val))
                    if len(key) > max_key:
                        max_key = len(key)
                else:
                    parts.append(None)
            # Rebuild aligned
            for idx, ln in enumerate(block):
                if parts[idx] is None:
                    out.append(ln)
                else:
                    key, val = parts[idx]
                    padding = ' ' * (max_key - len(key) + 1)  # at least one space after ':'
                    out.append(f"{key}:{padding}{val}")
            block.clear()

        for ln in lines:
            if ln.strip() == "":
                # blank -> flush current block and keep blank line
                flush_block()
                out.append(ln)
            else:
                block.append(ln)
        flush_block()
        return out

    def _align_colon_global(self, lines: List[str]) -> List[str]:
        """Align values after ':' so they start at the same column across all lines."""
        # Find max key length among all lines with ':'
        max_key = 0
        parsed: List[Optional[tuple]] = []
        for ln in lines:
            m = re.match(r"^(.*?):\s*(.*)$", ln)
            if m:
                key = m.group(1)
                val = m.group(2)
                parsed.append((key, val))
                if len(key) > max_key:
                    max_key = len(key)
            else:
                parsed.append(None)

        out: List[str] = []
        for idx, ln in enumerate(lines):
            if parsed[idx] is None:
                out.append(ln)
            else:
                key, val = parsed[idx]
                padding = ' ' * (max_key - len(key) + 1)
                out.append(f"{key}:{padding}{val}")
        return out

    def log_progress(self, current: int, total: int, description: str = "Progress", emoji: str = "📊"):
        """Log progress with consistent formatting."""
        logger = logging.getLogger()
        logger.info(f"\033[38;5;236m{emoji} {description}: {current}/{total}\033[0m")

    def log_completion(self, title: str, color: str = '[38;5;16m'):
        """Log completion status."""
        logger = logging.getLogger()
        logger.info(f"{color}✅ {title}[0m")

    def log_evaluation_result(self, checkpoint: str, metrics: Dict[str, float], sample_count: int):
        """Log evaluation results in structured format."""
        logger = logging.getLogger()
        
        acc = metrics.get('accuracy', 0.0)
        bal_acc = metrics.get('balanced_accuracy', 0.0)
        loss = metrics.get('loss', 0.0)
        
        logger.info(f"[38;5;16m📈 Accuracy: {acc:.4f}[0m")
        logger.info(f"[38;5;16m📈 Balanced Accuracy: {bal_acc:.4f}[0m")  
        logger.info(f"[38;5;17m📈 Evaluation Loss: {loss:.4f}[0m")
        logger.info(f"📊   Accuracy: {acc:.4f} ({acc*100:.2f}%) |   "
                   f"Balanced Accuracy: {bal_acc:.4f} ({bal_acc*100:.2f}%) |   "
                   f"Samples: {sample_count}")

    def log_training_metrics(self, epoch: int, phase: str, metrics: Dict[str, float], 
                           samples: Optional[int] = None, is_best: bool = False):
        """Log training metrics with structured format."""
        logger = logging.getLogger()
        
        acc = metrics.get('accuracy', 0.0)
        loss = metrics.get('loss', 0.0)
        
        # Format metrics display
        metrics_str = f"Loss: {loss:.4f}, Acc: {acc:.4f}"
        if samples:
            metrics_str += f", Samples: {samples}"
        
        # Add best marker
        best_marker = " ⭐ NEW BEST!" if is_best else ""
        
        logger.info(f"📊 Epoch {epoch:2d} [{phase:>5}] - {metrics_str}{best_marker}")

    def log_dataset_info(self, train_size: int, eval_size: int, num_checkpoints: int, checkpoint_list: list):
        """Log dataset information in structured box."""
        content = {
            "Training dataset size": train_size,
            "Evaluation dataset size": eval_size, 
            "Number of checkpoints": num_checkpoints,
            "Checkpoints": ", ".join(checkpoint_list) if len(checkpoint_list) <= 10 else f"{', '.join(checkpoint_list[:10])}, ..."
        }
        self.log_info_box("Dataset Information", content, 'cyan')

    def log_config_summary(self, config: Dict[str, Any]):
        """Log configuration summary in structured box."""
        # Extract key config info
        content = {
            "Device": config.get('device', 'unknown'),
            "Seed": config.get('seed', 'unknown'),
            "Validation Mode": 'balanced_acc',
            "Mode": config.get('mode', 'unknown'),
            "Config": config.get('config', 'unknown'),
            "Mode Type": self._extract_mode_type(config.get('config', '')),
            "Camera": config.get('camera', 'unknown')
        }
        self.log_info_box("Pipeline Configuration", content, 'cyan')

    def log_training_header(self, mode: str, epochs: int):
        """Log clean training header like the example."""
        logger = logging.getLogger()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"              🚀 Training {mode} - {epochs} epochs")
        logger.info("=" * 60)

    def log_training_epoch(self, epoch: int, phase: str, loss: float, acc: float, 
                          bal_acc: float, lr: float, samples: Optional[int] = None, 
                          is_best: bool = False, emoji: str = None):
        """Log training epoch in clean format matching the example."""
        logger = logging.getLogger()
        
        # Choose emoji based on phase or use custom emoji
        if emoji:
            selected_emoji = emoji
        elif phase.upper() == "TRAIN":
            selected_emoji = "🔹"
        elif phase.upper() == "VAL" or "VAL" in phase.upper():
            selected_emoji = "🔸"
        elif phase.upper() == "TEST" or "TEST" in phase.upper():
            selected_emoji = "🔻"
        else:
            selected_emoji = "📊"
        
        # Format metrics
        metrics_str = f"Loss: {loss:.4f} | Acc: {acc:.4f} | Bal.Acc: {bal_acc:.4f}"
        
        if lr is not None and (phase.upper() == "TRAIN" or "ROUND" in phase.upper()):
            metrics_str += f" | LR: {lr:.8f}"
        
        if samples is not None:
            metrics_str += f" | Samples: {samples}"
        
        if is_best:
            metrics_str += " ✓ BEST ACC (↑) SAVED"
        
        logger.info(f"{selected_emoji} Epoch {epoch:2d} [{phase.upper():>5}] {metrics_str}")

    def log_training_separator(self):
        """Log separator line for training epochs."""
        logger = logging.getLogger()
        logger.info("─" * 80)

    def log_training_summary(self, epoch: int, train_acc: float, train_bal_acc: float,
                           val_acc: float, val_bal_acc: float, val_loss: float, 
                           lr: float, status: str = ""):
        """Log epoch summary in clean format."""
        logger = logging.getLogger()
        
        summary = (f"Epoch {epoch}: train_acc={train_acc:.4f}, train_balanced_acc={train_bal_acc:.4f}, "
                  f"val_acc={val_acc:.4f}, val_balanced_acc={val_bal_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, LR={lr:.8f}")
        
        if status:
            summary += f" ({status})"
        
        logger.info(summary)

    def log_model_summary(self, model_type: str, num_classes: int, model_path: str = None):
        """Log concise model loading summary."""
        logger = logging.getLogger()
        
        if model_path:
            logger.info(f"🤖 Loaded {model_type} model ({num_classes} classes) from {model_path}")
        else:
            logger.info(f"🤖 Created {model_type} model ({num_classes} classes)")

    def log_class_distribution(self, class_stats: dict, total_train: int = None, total_val: int = None):
        """Log per-class distribution table."""
        logger = logging.getLogger()
        
        logger.info("ℹ️  📋 Per-class distribution:")
        logger.info("ℹ️  Class                Train    Val      Val%    ")
        logger.info("ℹ️  " + "─" * 54)
        
        total_train_count = 0
        total_val_count = 0
        
        for class_name, stats in class_stats.items():
            train_count = stats.get('train', 0)
            val_count = stats.get('val', 0)
            val_percent = (val_count / train_count * 100) if train_count > 0 else 0
            
            total_train_count += train_count
            total_val_count += val_count
            
            logger.info(f"ℹ️  {class_name:<20} {train_count:<8} {val_count:<8} {val_percent:<7.1f} %")
        
        logger.info("ℹ️  " + "─" * 54)
        total_val_percent = (total_val_count / total_train_count * 100) if total_train_count > 0 else 0
        logger.info(f"ℹ️  TOTAL                {total_train_count:<8} {total_val_count:<8} {total_val_percent:<7.1f} %")

    def log_training_completion(self, mode: str):
        """Log training completion."""
        logger = logging.getLogger()
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"              ✅ Training {mode} completed successfully")
        logger.info("=" * 60)
        logger.info("")

    def log_model_info(self, message: str):
        """Log model information with timestamp."""
        logger = logging.getLogger()
        logger.info(message)

    # ========== NEW STRUCTURED LOGGING METHODS ==========
    
    def log_initial_setup(self, experiment_dir: str, log_file: str, seed: int, 
                         camera: str, num_checkpoints: int, train_path: str, test_path: str):
        """Log initial setup information - now merged into setup_details."""
        # This function is now deprecated and does nothing
        # All logging is handled by log_setup_details
        pass
    
    def log_setup_details(self, camera: str, log_location: str, model_info: dict, config_dict: dict, 
                         num_checkpoints: int = None, train_path: str = None, test_path: str = None):
        """Log setup details - Phase 1 of logging structure with merged initial setup."""
        # Compose a single boxed section containing all setup details
        setup_lines = [
            f"Camera: {camera}",
            f"Ckpt: {num_checkpoints if num_checkpoints is not None else 'auto'}",
            f"Train Path: {train_path if train_path else 'auto-detected'}",
            f"Test Path: {test_path if test_path else 'auto-detected'}",
            f"Log Dir: {log_location}",
            f"Random Seed: {config_dict.get('seed', 42)}",
        ]

        model_name = model_info.get('name', 'BioCLIP')
        model_source = model_info.get('source', 'pre-trained (original)')
        model_classes = model_info.get('num_classes', 'auto-detected')
        model_lines = [
            f"Model: {model_name}",
            f"Source: {model_source}",
            f"Classes: {model_classes}",
        ]

        config_lines = self._build_config_lines(config_dict)

        all_lines = setup_lines + [""] + model_lines + [""] + config_lines
        self.log_box("🚀 CAMERA TRAP FRAMEWORK V2 - SETUP DETAILS", all_lines, 'cyan')
    
    def _log_config_table(self, config: dict):
        """Log configuration in a boxed format (no 'Parameter: Value' headers)."""
        lines = self._build_config_lines(config)
        self.log_box("Configuration Details", lines, 'cyan')

    def _build_config_lines(self, config: dict) -> List[str]:
        """Build configuration lines for boxed rendering."""
        lines: List[str] = [
            f"Mode: {config.get('mode', 'train')}",
            f"Device: {config.get('system', {}).get('device', 'cuda')}",
            f"Seed: {config.get('system', {}).get('seed', 42)}",
            f"Config File: {config.get('config', 'unknown')}",
            f"Model Version: {config.get('model', {}).get('version', 'v2')}",
        ]

        if 'training' in config:
            training = config['training']
            lines.extend([
                f"Epochs: {training.get('epochs', 30)}",
                f"Train Batch Size: {training.get('train_batch_size', 128)}",
                f"Eval Batch Size: {training.get('eval_batch_size', 512)}",
                f"Learning Rate: {training.get('optimizer_params', {}).get('lr', 0.0001)}",
                f"Optimizer: {training.get('optimizer_name', 'AdamW')}",
                f"Scheduler: {training.get('scheduler', 'CosineAnnealingLR')}",
            ])

        if 'model' in config:
            model = config['model']
            lines.extend([
                f"Pretrained: {'True' if model.get('pretrained', True) else 'False'}",
                f"Use PEFT: {'True' if model.get('use_peft', False) else 'False'}",
                f"Freeze Backbone: {'True' if model.get('freeze_backbone', False) else 'False'}",
            ])

        return lines
    
    def log_dataset_details(self, train_size: int, test_size: int, num_checkpoints: int, 
                          num_classes: int, class_distribution: dict, checkpoint_list: list = None):
        """Log dataset details - Phase 2 of logging structure."""
        logger = logging.getLogger()

        logger.info("=" * 80)
        logger.info("📊 DATASET DETAILS")
        logger.info("=" * 80)
        logger.info("")

        # Build combined lines for the box
        dataset_lines = [
            f"Training Size: {train_size:,}",
            f"Test Size: {test_size:,}",
            f"Ckpt: {num_checkpoints}",
            f"Classes: {num_classes}",
            "<hr>",
        ] + self._build_dataset_overview_lines(class_distribution)

        self.log_box("Dataset Information", dataset_lines, 'cyan')
        logger.info("")
    
    def log_dataset_overview(self, class_distribution: dict):
        """Log simple dataset overview showing only Train and Test columns."""
        logger = logging.getLogger()
        
        logger.info("ℹ️  📋 Per-class distribution:")
        logger.info("ℹ️  Class                Train    Test     ")
        logger.info("ℹ️  " + "─" * 40)
        
        total_train = 0
        total_test = 0
        
        # Sort classes by training count (descending)
        sorted_classes = sorted(class_distribution.items(), 
                              key=lambda x: x[1].get('train', 0), reverse=True)
        
        for class_name, stats in sorted_classes:
            train_count = stats.get('train', 0)
            test_count = stats.get('test', 0)
            
            total_train += train_count
            total_test += test_count
            
            # Truncate class name if too long
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            
            logger.info(f"ℹ️  {display_name:<20} {train_count:<8} {test_count:<8}")
        
        logger.info("ℹ️  " + "─" * 40)
        logger.info(f"ℹ️  TOTAL                {total_train:<8} {total_test:<8}")

    def _build_dataset_overview_lines(self, class_distribution: dict) -> List[str]:
        """Return lines representing the per-class Train/Test table for embedding in a box."""
        lines: List[str] = []
        header_prefix = ""
        lines.append(f"{header_prefix}Per-class distribution:")
        lines.append(f"{header_prefix}Class                Train    Test")
        lines.append("─" * 40)

        total_train = 0
        total_test = 0

        sorted_classes = sorted(class_distribution.items(), key=lambda x: x[1].get('train', 0), reverse=True)
        for class_name, stats in sorted_classes:
            train_count = stats.get('train', 0)
            test_count = stats.get('test', 0)
            total_train += train_count
            total_test += test_count
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            # Keep spacing similar to outer printing
            lines.append(f"{display_name:<20} {train_count:<8} {test_count:<8}")

        lines.append("─" * 40)
        lines.append(f"TOTAL                {total_train:<8} {total_test:<8}")
        return lines

    def log_oracle_validation_distribution(self, class_distribution: dict):
        """Log Oracle validation strategy showing Train, Test, Val columns with (-number) indicators."""
        logger = logging.getLogger()
        
        logger.info("ℹ️  📋 Oracle Validation Strategy (2 samples per class):")
        logger.info("ℹ️  Class                Train    Test     Val      Val%    ")
        logger.info("ℹ️  " + "─" * 60)
        
        total_train = 0
        total_test = 0
        total_val = 0
        
        # Sort classes by training count (descending)
        sorted_classes = sorted(class_distribution.items(), 
                              key=lambda x: x[1].get('original_train', 0), reverse=True)
        
        for class_name, stats in sorted_classes:
            original_train = stats.get('original_train', 0)
            test_count = stats.get('test', 0)
            val_count = stats.get('val', 0)
            val_from_train = stats.get('val_from_train', 0)
            val_from_test = stats.get('val_from_test', 0)
            
            # Calculate final train count after validation removal
            final_train = original_train - val_from_train
            
            total_train += final_train
            total_test += test_count
            total_val += val_count
            
            val_percent = (val_count / original_train * 100) if original_train > 0 else 0
            
            # Truncate class name if too long
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            
            # Add indicators for where validation samples came from
            train_display = str(final_train)
            test_display = str(test_count)
            
            if val_from_train > 0:
                train_display += f"(-{val_from_train})"
            if val_from_test > 0:
                test_display += f"(-{val_from_test})"
            
            logger.info(f"ℹ️  {display_name:<20} {train_display:<8} {test_display:<8} {val_count:<8} {val_percent:<7.1f} %")
        
        logger.info("ℹ️  " + "─" * 60)
        total_val_percent = (total_val / (total_train + total_val) * 100) if (total_train + total_val) > 0 else 0
        logger.info(f"ℹ️  TOTAL                {total_train:<8} {total_test:<8} {total_val:<8} {total_val_percent:<7.1f} %")

    def log_accumulative_checkpoint_validation(self, checkpoint: str, class_distribution: dict):
        """Log Accumulative validation for specific checkpoint (using that checkpoint's test data as validation)."""
        logger = logging.getLogger()
        
        logger.info(f"ℹ️  📋 Accumulative Training {checkpoint} - Validation Strategy:")
        logger.info("ℹ️  Class                Train    Val      Val%    ")
        logger.info("ℹ️  " + "─" * 54)
        
        total_train = 0
        total_val = 0
        
        # Sort classes by training count (descending)
        sorted_classes = sorted(class_distribution.items(), 
                              key=lambda x: x[1].get('train', 0), reverse=True)
        
        for class_name, stats in sorted_classes:
            train_count = stats.get('train', 0)
            val_count = stats.get('val', 0)
            val_percent = (val_count / train_count * 100) if train_count > 0 else 0
            
            total_train += train_count
            total_val += val_count
            
            # Truncate class name if too long
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            
            logger.info(f"ℹ️  {display_name:<20} {train_count:<8} {val_count:<8} {val_percent:<7.1f} %")
        
        logger.info("ℹ️  " + "─" * 54)
        total_val_percent = (total_val / train_count * 100) if train_count > 0 else 0
        logger.info(f"ℹ️  TOTAL                {total_train:<8} {total_val:<8} {total_val_percent:<7.1f} %")
        logger.info(f"ℹ️  Note: Validation samples are from {checkpoint} test data")

    def _log_class_distribution_table(self, class_distribution: dict):
        """Log per-class distribution in the exact format requested."""
        logger = logging.getLogger()
        
        logger.info("ℹ️  📋 Per-class distribution:")
        logger.info("ℹ️  Class                Train    Val      Val%    ")
        logger.info("ℹ️  " + "─" * 54)
        
        total_train = 0
        total_val = 0
        
        # Sort classes by training count (descending)
        sorted_classes = sorted(class_distribution.items(), 
                              key=lambda x: x[1].get('train', 0), reverse=True)
        
        for class_name, stats in sorted_classes:
            train_count = stats.get('train', 0)
            val_count = stats.get('val', 0)
            val_percent = (val_count / train_count * 100) if train_count > 0 else 0
            
            total_train += train_count
            total_val += val_count
            
            # Truncate class name if too long
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            
            logger.info(f"ℹ️  {display_name:<20} {train_count:<8} {val_count:<8} {val_percent:<7.1f} %")
        
        logger.info("ℹ️  " + "─" * 54)
        total_val_percent = (total_val / total_train * 100) if total_train > 0 else 0
        logger.info(f"ℹ️  TOTAL                {total_train:<8} {total_val:<8} {total_val_percent:<7.1f} %")
    
    def log_accumulative_round_distribution(self, checkpoint_round: int, class_distribution: dict):
        """Log per-class distribution for accumulative training round with Train/Val/Test columns."""
        logger = logging.getLogger()
        
        logger.info(f"ℹ️  📋 Round {checkpoint_round} - Per-class distribution:")
        logger.info("ℹ️  Class                Train    Val      Test    ")
        logger.info("ℹ️  " + "─" * 48)
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        # Sort classes by training count (descending)
        sorted_classes = sorted(class_distribution.items(), 
                              key=lambda x: x[1].get('train', 0), reverse=True)
        
        for class_name, stats in sorted_classes:
            train_count = stats.get('train', 0)
            val_count = stats.get('val', 0)
            test_count = stats.get('test', 0)
            
            total_train += train_count
            total_val += val_count
            total_test += test_count
            
            # Truncate class name if too long
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            
            logger.info(f"ℹ️  {display_name:<20} {train_count:<8} {val_count:<8} {test_count:<8}")
        
        logger.info("ℹ️  " + "─" * 48)
        logger.info(f"ℹ️  TOTAL                {total_train:<8} {total_val:<8} {total_test:<8}")

    def log_oracle_class_distribution(self, class_distribution: dict):
        """Log per-class distribution for Oracle training showing Train/Val/Test with plain numbers."""
        logger = logging.getLogger()
        
        logger.info("ℹ️  📋 Per-class distribution:")
        logger.info("ℹ️  Class                Train    Val      Test    ")
        logger.info("ℹ️  " + "─" * 48)
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        # Sort classes alphabetically for consistent output
        for class_name in sorted(class_distribution.keys()):
            counts = class_distribution[class_name]
            train_count = counts['train']
            val_count = counts['val']
            test_count = counts['test']
            
            total_train += train_count
            total_val += val_count
            total_test += test_count
            
            # Truncate class name if too long
            display_name = class_name[:20] if len(class_name) <= 20 else class_name[:17] + "..."
            
            logger.info(f"ℹ️  {display_name:<20} {train_count:<8} {val_count:<8} {test_count:<8}")
        
        logger.info("ℹ️  " + "─" * 48)
        logger.info(f"ℹ️  TOTAL                {total_train:<8} {total_val:<8} {total_test:<8}")

    def log_training_phase_header(self, mode: str, epochs: int, num_classes: int = None, 
                                num_train_checkpoints: int = None, num_test_checkpoints: int = None, 
                                total_samples: int = None):
        """Log training phase header - Phase 3 of logging structure."""
        logger = logging.getLogger()
        
        logger.info("=" * 80)
        logger.info("🏋️  TRAINING PHASE")
        logger.info("=" * 80)
        logger.info("")
        
        # Training information in table format
        training_configs = [
            ("Mode", f"{mode}"),
            ("Epochs", epochs),
        ]
        
        if num_classes:
            training_configs.append(("Classes", num_classes))
        
        if num_train_checkpoints:
            training_configs.append(("Train Ckpt", num_train_checkpoints))
            
        if num_test_checkpoints:
            training_configs.append(("Test Ckpt", num_test_checkpoints))
            
        if total_samples:
            training_configs.append(("Total Samples", f"{total_samples:,}"))
        
        # Fixed column widths matching the style
        key_width = 16
        val_width = 15
        
        logger.info("🚀 Training Information:")
        logger.info(f"   {'Parameter':<{key_width}} : {'Value':<{val_width}}")
        logger.info(f"   {'-' * key_width} : {'-' * val_width}")
        
        for key, value in training_configs:
            logger.info(f"   {key:<{key_width}} : {value:<{val_width}}")
        
        logger.info("")
    
    def log_testing_phase_header(self, num_checkpoints: int):
        """Log testing phase header - Phase 4 of logging structure."""
        logger = logging.getLogger()
        
        logger.info("=" * 80)
        logger.info("🧪 TESTING PHASE")
        logger.info("=" * 80)
        logger.info("")
        logger.info(f"📊 Evaluating model on {num_checkpoints} checkpoints...")
        logger.info("")
    
    def log_final_summary(self, results_summary: dict):
        """Log final overall result summary - Phase 5 of logging structure."""
        logger = logging.getLogger()
        
        logger.info("=" * 80)
        logger.info("🎯 FINAL RESULTS SUMMARY")
        logger.info("=" * 80)
        logger.info("")
        
        # Main metrics
        avg_acc = results_summary.get('average_accuracy', 0.0)
        avg_bal_acc = results_summary.get('average_balanced_accuracy', 0.0)
        num_checkpoints = results_summary.get('num_checkpoints', 0)
        
        logger.info(f"📊 Checkpoints Evaluated: {num_checkpoints}")
        logger.info(f"📈 Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        logger.info(f"📈 Average Balanced Accuracy: {avg_bal_acc:.4f} ({avg_bal_acc*100:.2f}%)")
        
        # Performance assessment
        if avg_bal_acc >= 0.8:
            performance = "🌟 Excellent"
        elif avg_bal_acc >= 0.6:
            performance = "✅ Good"
        elif avg_bal_acc >= 0.4:
            performance = "⚠️  Fair"
        else:
            performance = "❌ Needs Improvement"
        
        logger.info(f"🎯 Performance Level: {performance}")
        logger.info("")
        
        # Additional details if available
        if 'best_checkpoint' in results_summary:
            best_ckp = results_summary['best_checkpoint']
            best_acc = results_summary.get('best_accuracy', 0.0)
            logger.info(f"🏆 Best Checkpoint: {best_ckp} ({best_acc:.4f})")
        
        if 'worst_checkpoint' in results_summary:
            worst_ckp = results_summary['worst_checkpoint']
            worst_acc = results_summary.get('worst_accuracy', 0.0)
            logger.info(f"📉 Worst Checkpoint: {worst_ckp} ({worst_acc:.4f})")
        
        logger.info("")
        logger.info("🎉 Framework execution completed successfully!")
        logger.info("=" * 80)

    def log_execution_summary(self, mode_type: str, checkpoint_count: int, 
                            log_path: str, results_path: str):
        """Log execution completion summary."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = {
            "Mode": mode_type.upper(),
            "Total Checkpoints": checkpoint_count,
            "Logs": log_path,
            "Results": results_path,
            "Completed": timestamp
        }
        self.log_info_box("Execution Summary", content, 'cyan')

    def _extract_mode_type(self, config_path: str) -> str:
        """Extract mode type from config path."""
        if not config_path:
            return 'unknown'
        
        if 'zs.yaml' in config_path:
            return 'zs'
        elif 'oracle.yaml' in config_path:
            return 'oracle'
        elif 'accumulative.yaml' in config_path:
            return 'accumulative'
        else:
            return Path(config_path).stem


# Global logger instance
icicle_logger = ICICLELogger()

# Convenience functions for backward compatibility
def setup_logging(output_dir: str, debug: bool = False, experiment_name: str = "experiment", use_timestamps: bool = False) -> logging.Logger:
    """Setup logging wrapper function."""
    return icicle_logger.setup_logging(output_dir, debug, experiment_name, use_timestamps)
