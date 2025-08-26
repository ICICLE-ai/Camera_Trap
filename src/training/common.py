"""
Common training/evaluation utilities.

This module centralizes shared logic used by oracle and accumulative training
to keep main.py small and avoid code duplication.
"""

from typing import Dict, Tuple, List, Optional
import logging

import torch
import torch.nn as nn
import numpy as np

from src.utils import icicle_logger
from src.utils.metrics import MetricsCalculator
from src.utils.paths import (
	get_checkpoint_directories,
	load_checkpoint_data,
)
from src.models.factory import create_model
from src.data.dataset import get_dataloaders


logger = logging.getLogger(__name__)


def evaluate_epoch(model: torch.nn.Module,
				   data_loader,
				   criterion: nn.Module,
				   device: torch.device,
				   ) -> Tuple[float, float, float, int]:
	"""Evaluate model for one epoch over a dataloader.

	Returns (avg_loss, accuracy, balanced_accuracy, total_samples).
	"""
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	all_predictions: List[int] = []
	all_labels: List[int] = []

	with torch.no_grad():
		for batch_idx, batch in enumerate(data_loader):
			images = batch['image'].to(device)
			labels = batch['label'].to(device)

			outputs = model(images)
			loss = criterion(outputs, labels)

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			all_predictions.extend(predicted.cpu().numpy())
			all_labels.extend(labels.cpu().numpy())

			# free ASAP
			del images, labels, outputs, predicted
			if torch.cuda.is_available() and (batch_idx % 10 == 0):
				torch.cuda.empty_cache()
				if hasattr(torch.cuda, 'ipc_collect'):
					torch.cuda.ipc_collect()

	avg_loss = running_loss / len(data_loader) if len(data_loader) > 0 else 0.0
	accuracy = (correct / total) if total > 0 else 0.0

	if len(all_predictions) > 0:
		all_predictions_np = np.array(all_predictions)
		all_labels_np = np.array(all_labels)
		# compute balanced accuracy via MetricsCalculator to stay consistent
		n_classes = int(np.max(all_labels_np)) + 1
		class_names = [str(i) for i in range(n_classes)]
		mc = MetricsCalculator(class_names)
		m = mc.calculate_metrics(all_predictions_np, all_labels_np)
		bal_acc = float(m['balanced_accuracy'])
	else:
		bal_acc = 0.0

	# cleanup
	del all_predictions, all_labels
	import gc
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return avg_loss, accuracy, bal_acc, total


def setup_model_and_data(config: Dict, args, mode: str = 'oracle', current_checkpoint: Optional[str] = None):
	"""Create model and dataloaders according to the requested mode."""
	model = create_model(config)
	train_loader, val_loader, test_loader = get_dataloaders(
		config, mode=mode, current_checkpoint=current_checkpoint, verbose=False
	)
	return model, train_loader, val_loader, test_loader


def _build_simple_dataset(samples: List[Dict], class_to_idx: Dict[str, int]):
	"""Create a lightweight torch Dataset for a flat list of samples."""
	from torch.utils.data import Dataset
	from PIL import Image
	import torchvision.transforms as transforms

	transform = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	class SimpleDataset(Dataset):
		def __init__(self, samples, class_to_idx):
			self.samples = samples
			self.class_to_idx = class_to_idx

		def __len__(self):
			return len(self.samples)

		def __getitem__(self, idx):
			sample = self.samples[idx]
			try:
				image = Image.open(sample['image_path']).convert('RGB')
				image = transform(image)
			except Exception as e:
				logger.warning(f"Could not load image {sample.get('image_path','?')}: {e}")
				image = torch.randn(3, 224, 224)
			label = self.class_to_idx[sample['common']]
			return {
				'image': image,
				'label': label,
				'image_path': sample.get('image_path', ''),
				'common_name': sample.get('common', '')
			}

	return SimpleDataset(samples, class_to_idx)


def evaluate_checkpoints(config: Dict,
						 args,
						 trained_model: Optional[torch.nn.Module] = None,
						 ) -> Tuple[float, Dict[str, Dict]]:
	"""Evaluate a model per checkpoint on the test.json file.

	Returns (avg_balanced_accuracy, checkpoint_results_dict).
	"""
	import torch
	from torch.utils.data import DataLoader

	logger.info("Starting per-checkpoint evaluation…")
	checkpoints = get_checkpoint_directories(args.camera)
	if not checkpoints:
		logger.error(f"No checkpoints found for camera {args.camera}")
		return 0.0, {}

	# Prepare model
	model = trained_model if trained_model is not None else create_model(config)
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	criterion = nn.CrossEntropyLoss()

	# Load test data once
	test_path = config.get('data', {}).get('test_path')
	test_data = load_checkpoint_data(test_path) if test_path else {}

	# Build class mapping
	class_names = config.get('data', {}).get('class_names') or []
	if not class_names:
		# derive from data
		classes = set()
		for ckp, samples in test_data.items():
			if ckp.startswith('ckp_'):
				for s in samples:
					classes.add(s['common'])
		class_names = sorted(list(classes))
	class_to_idx = {c: i for i, c in enumerate(class_names)}

	# Use a conservative default eval batch size to avoid VRAM spikes
	eval_bs = int(config.get('training', {}).get('eval_batch_size', 8))

	results: Dict[str, Dict] = {}
	all_bal_acc: List[float] = []
	all_acc: List[float] = []

	for i, ckp in enumerate(checkpoints, 1):
		icicle_logger.log_progress(i, len(checkpoints), f"Evaluating {ckp}")
		samples = test_data.get(ckp, [])
		if not samples:
			logger.warning(f"No samples for {ckp}")
			continue

		ds = _build_simple_dataset(samples, class_to_idx)
		dl = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=False)

		loss, acc, bal_acc, total = evaluate_epoch(model, dl, criterion, device)
		# Print per-checkpoint quick metric summary with clear indenting
		try:
			# Two-level indentation for readability under the progress line
			label = "balanced acc:"
			icicle_logger.log_model_info(f"      {label:<15} {bal_acc:>10.4f}")
		except Exception:
			pass
		metrics = {
			'accuracy': float(acc),
			'balanced_accuracy': float(bal_acc),
			'loss': float(loss),
		}

		results[ckp] = {'metrics': metrics, 'sample_count': int(total)}
		all_bal_acc.append(bal_acc)
		all_acc.append(acc)

		# Proactive GPU memory cleanup between checkpoints
		try:
			import gc
			del dl, ds
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				# Collect IPC memory fragments (helps on some drivers)
				if hasattr(torch.cuda, 'ipc_collect'):
					torch.cuda.ipc_collect()
		except Exception:
			pass

	avg_bal_acc = float(np.mean(all_bal_acc)) if all_bal_acc else 0.0
	return avg_bal_acc, results

