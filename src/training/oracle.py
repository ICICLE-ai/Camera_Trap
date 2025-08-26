"""
Oracle training strategy implementation.

Trains on all available training data while holding out a small validation set
per-class and evaluates per checkpoint averaged across test checkpoints.
"""

from typing import Dict, Tuple
from copy import deepcopy
import numpy as np
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

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
	train_cfg = config.get('training', {})
	opt_params = (train_cfg.get('optimizer_params') or {})
	lr = float(opt_params.get('lr', train_cfg.get('learning_rate', 1e-4)))
	wd = float(opt_params.get('weight_decay', train_cfg.get('weight_decay', 1e-4)))
	optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

	# Determine base epochs before scheduler setup
	base_epochs = args.epochs or config.get('training', {}).get('epochs', 30)

	# Optional LR scheduler (CosineAnnealingLR)
	scheduler = None
	sch_name = (train_cfg.get('scheduler') or '').lower()
	if sch_name == 'cosineannealinglr':
		sp = train_cfg.get('scheduler_params', {}) or {}
		T_max = int(sp.get('T_max', base_epochs))
		eta_min = float(sp.get('eta_min', lr * 0.1))
		scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

	# Use standard FP32 training (previous behavior)
	use_cuda = torch.cuda.is_available()
	# Early stopping policy when validation is enabled
	use_es = bool(args.train_val and val_loader and len(val_loader) > 0)
	target_epochs = base_epochs if not use_es else 10

	val_ba_hist = []  # track validation balanced accuracy per epoch
	best_val_ba = float('-inf')
	best_epoch = -1
	best_state = None
	epoch = 0
	while epoch < target_epochs:
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
			improved = v_ba > best_val_ba
			val_line = f"🔸 Epoch {epoch:2d} [ VAL ] Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Bal.Acc: {v_ba:.4f}"
			if improved:
				val_line += "  ✓ BEST ACC (↑) SAVED"
			print(val_line)
			if improved:
				best_val_ba = float(v_ba)
				best_epoch = int(epoch)
				best_state = deepcopy(model.state_dict())
			if use_es:
				val_ba_hist.append(float(v_ba))

		if args.train_test:
			t_loss, t_acc, t_ba, t_n = evaluate_oracle_per_checkpoint(model, config, criterion, device)
			print(f"🔻 Epoch {epoch:2d} [TEST*] Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | Bal.Acc: {t_ba:.4f}")

		# Visual separator between epochs
		print("" + "────────────────────────────────────────────────────────────────────────────────")

		# Early stopping decision point after 10 epochs (warmup=5, monitor=5)
		if use_es and epoch == 9:  # decision after completing 10 epochs (0-based)
			if len(val_ba_hist) >= 10:
				best_idx_10 = int(np.argmax(val_ba_hist[:10])) + 1
				best_val_10 = float(val_ba_hist[best_idx_10 - 1])
				if best_idx_10 == 10:
					icicle_logger.log_model_info(
						f"Early stopping: improvement at epoch 10; continuing up to {min(base_epochs, 20)} epochs."
					)
					target_epochs = max(target_epochs, min(base_epochs, 20))
				else:
					best_epoch = best_idx_10 - 1
					best_val_ba = best_val_10
					icicle_logger.log_model_info("Early stopping triggered after 10 epochs (warmup=5, monitor=5)")
					icicle_logger.log_model_info(f"Best epoch within first 10: {best_idx_10} with val_balanced_acc={best_val_10:.4f}")
					break

		# Step scheduler at end of epoch
		if scheduler is not None:
			scheduler.step()

		epoch += 1

	icicle_logger.log_training_completion("oracle")

	# If we tracked a best epoch on validation, use those weights for saving and show best-epoch recap
	if best_state is not None and best_epoch >= 0:
		try:
			model.load_state_dict(best_state)
			icicle_logger.log_model_info(
				f"Selected best epoch {best_epoch + 1} (val_balanced_acc={best_val_ba:.4f}); using these weights for final testing/saving."
			)
			# End-of-run summary with best weights (multiline, aligned)
			val_metrics = None
			test_metrics = None
			if args.train_val and val_loader and len(val_loader) > 0:
				v_loss, v_acc, v_ba, v_n = evaluate_epoch(model, val_loader, criterion, device)
				val_metrics = (v_loss, v_acc, v_ba)
			if args.train_test:
				t_loss, t_acc, t_ba, t_n = evaluate_oracle_per_checkpoint(model, config, criterion, device)
				test_metrics = (t_loss, t_acc, t_ba)
			if val_metrics or test_metrics:
				lines = [f"Best epoch {best_epoch + 1} —"]
				key_w = 7
				if val_metrics:
					vl, va, vba = val_metrics
					lines += [
						"  VAL:",
						f"    {'Loss':<{key_w}}: {vl:.4f}",
						f"    {'Acc':<{key_w}}: {va:.4f}",
						f"    {'Bal.Acc':<{key_w}}: {vba:.4f}",
					]
				if test_metrics:
					tl, ta, tba = test_metrics
					lines += [
						"  TEST*:",
						f"    {'Loss':<{key_w}}: {tl:.4f}",
						f"    {'Acc':<{key_w}}: {ta:.4f}",
						f"    {'Bal.Acc':<{key_w}}: {tba:.4f}",
					]
				icicle_logger.log_model_info("\n".join(lines))
		except Exception:
			pass

	model_path = 'oracle_model.pth'
	try:
		torch.save(model.state_dict(), model_path)
		logger.info(f"Model saved to {model_path}")
	except Exception as e:
		logger.warning(f"Failed to save model: {e}")

	return model

