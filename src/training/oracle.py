"""
Oracle training strategy implementation.

Trains on all available training data while holding out a small validation set
per-class and evaluates per checkpoint averaged across test checkpoints.
"""

from typing import Dict, Tuple
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from .common import (
	setup_model_and_data,
	evaluate_epoch,
	evaluate_checkpoints,
)
from src.utils import icicle_logger
from src.utils.paths import load_checkpoint_data


logger = logging.getLogger(__name__)


def evaluate_oracle_per_checkpoint(model, config: Dict, criterion, device) -> Tuple[float, float, float, int]:
	"""Evaluate Oracle model on each checkpoint separately and average results."""
	import json
	import numpy as np
	from torch.utils.data import DataLoader
	from .common import _build_simple_dataset

	data_config = config.get('data', {})
	test_path = data_config.get('test_path')
	with open(test_path, 'r') as f:
		test_data = json.load(f)

	class_names = config['data']['class_names']
	class_to_idx = {n: i for i, n in enumerate(class_names)}

	# Conservative eval batch size to reduce VRAM
	eval_bs = int(config.get('training', {}).get('eval_batch_size', 8))

	results = []
	total_samples = 0
	for ckp_key, samples in test_data.items():
		if not ckp_key.startswith('ckp_'):
			continue

		ds = _build_simple_dataset(samples, class_to_idx)
		dl = DataLoader(ds, batch_size=eval_bs, shuffle=False, num_workers=0, pin_memory=False)
		l, a, ba, n = evaluate_epoch(model, dl, criterion, device)
		results.append((l, a, ba))
		total_samples += n

		# Cleanup between checkpoints
		try:
			import gc
			del dl, ds
			gc.collect()
			if torch.cuda.is_available():
				torch.cuda.empty_cache()
				if hasattr(torch.cuda, 'ipc_collect'):
					torch.cuda.ipc_collect()
		except Exception:
			pass

	if results:
		import numpy as np
		arr = np.array(results)
		return float(arr[:, 0].mean()), float(arr[:, 1].mean()), float(arr[:, 2].mean()), int(total_samples)
	return 0.0, 0.0, 0.0, 0


def train(config: Dict, args) -> torch.nn.Module:
	"""Run Oracle training and return the trained model."""
	logger.info("Starting oracle training…")
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	# Sanity check data files
	data_cfg = config.get('data', {})
	train_path = data_cfg.get('train_path')
	test_path = data_cfg.get('test_path')
	if not train_path or not Path(train_path).exists():
		logger.error(f"Training data path not found: {train_path}")
		return None
	if not test_path or not Path(test_path).exists():
		logger.error(f"Test data path not found: {test_path}")
		return None

	# Build union class set from train/test to set num_classes
	train_data = load_checkpoint_data(train_path)
	test_data = load_checkpoint_data(test_path)
	classes = set()
	for d in (train_data, test_data):
		for k, samples in d.items():
			if k.startswith('ckp_'):
				for s in samples:
					classes.add(s['common'])
	class_names = sorted(list(classes))
	num_classes = len(class_names)
	config.setdefault('model', {})['num_classes'] = num_classes
	config.setdefault('data', {})['class_names'] = class_names

	# Dataloaders with oracle strategy
	model, train_loader, val_loader, test_loader = setup_model_and_data(config, args, mode='oracle')
	model = model.to(device)

	icicle_logger.log_training_phase_header(
		mode='oracle training',
		epochs=(args.epochs or config.get('training', {}).get('epochs', 30)),
		num_classes=num_classes,
		total_samples=len(train_loader.dataset) if train_loader else 0,
	)

	criterion = nn.CrossEntropyLoss()
	lr = config.get('training', {}).get('learning_rate', 1e-4)
	wd = config.get('training', {}).get('weight_decay', 1e-4)
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

	# Use standard FP32 training (previous behavior)
	use_cuda = torch.cuda.is_available()

	epochs = args.epochs or config.get('training', {}).get('epochs', 30)
	for epoch in range(epochs):
		model.train()
		running_loss = 0.0
		correct = 0
		total = 0
		for batch_idx, batch in enumerate(train_loader):
			images = batch['image'].to(device)
			labels = batch['label'].to(device)

			optimizer.zero_grad(set_to_none=True)
			outputs = model(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			_, predicted = outputs.max(1)
			total += labels.size(0)
			correct += predicted.eq(labels).sum().item()

			# Free tensors and periodically clear cache
			del images, labels, outputs, predicted
			if use_cuda and (batch_idx % 10 == 0):
				torch.cuda.empty_cache()
				if hasattr(torch.cuda, 'ipc_collect'):
					torch.cuda.ipc_collect()

		train_loss = running_loss / max(len(train_loader), 1)
		train_acc = (correct / total) if total > 0 else 0.0
		print(f"🔹 Epoch {epoch:2d} [TRAIN] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal.Acc: {train_acc:.4f}")

		if args.train_val and val_loader and len(val_loader) > 0:
			v_loss, v_acc, v_ba, v_n = evaluate_epoch(model, val_loader, criterion, device)
			print(f"🔸 Epoch {epoch:2d} [ VAL ] Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Bal.Acc: {v_ba:.4f}")

		if args.train_test:
			t_loss, t_acc, t_ba, t_n = evaluate_oracle_per_checkpoint(model, config, criterion, device)
			print(f"🔻 Epoch {epoch:2d} [TEST*] Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | Bal.Acc: {t_ba:.4f}")

	icicle_logger.log_training_completion("oracle")

	model_path = 'oracle_model.pth'
	try:
		torch.save(model.state_dict(), model_path)
		logger.info(f"Model saved to {model_path}")
	except Exception as e:
		logger.warning(f"Failed to save model: {e}")

	return model

