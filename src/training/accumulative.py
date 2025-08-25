"""
Accumulative training strategy implementation.

Progressively trains from ckp_1 → ckp_N-1 and validates on current ckp test,
testing on the next checkpoint each round.
"""

from typing import Dict
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import icicle_logger
from src.utils.paths import load_checkpoint_data, get_checkpoint_directories
from src.models.factory import create_model
from src.data.dataset import get_dataloaders
from .common import evaluate_epoch


logger = logging.getLogger(__name__)


def train(config: Dict, args) -> torch.nn.Module:
	"""Run accumulative training and return final model from the last round."""
	device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

	data_cfg = config.get('data', {})
	train_path = data_cfg.get('train_path')
	test_path = data_cfg.get('test_path')
	if not train_path or not Path(train_path).exists():
		logger.error(f"Training data path not found: {train_path}")
		return None
	if not test_path or not Path(test_path).exists():
		logger.error(f"Test data path not found: {test_path}")
		return None

	train_data = load_checkpoint_data(train_path)
	test_data = load_checkpoint_data(test_path)

	# Build class names
	classes = set()
	for d in (train_data, test_data):
		for k, samples in d.items():
			if k.startswith('ckp_'):
				for s in samples:
					classes.add(s['common'])
	class_names = sorted(list(classes))
	config.setdefault('model', {})['num_classes'] = len(class_names)
	config.setdefault('data', {})['class_names'] = class_names

	train_ckps = sorted([k for k in train_data.keys() if k.startswith('ckp_')], key=lambda x: int(x.split('_')[1]))
	test_ckps = sorted([k for k in test_data.keys() if k.startswith('ckp_')], key=lambda x: int(x.split('_')[1]))
	if not train_ckps:
		logger.error("No training checkpoints found in data")
		return None

	max_train_round = len(train_ckps) - 1  # leave last for next checkpoint testing

	epochs_per_round = args.epochs or config.get('training', {}).get('epochs', 30)
	icicle_logger.log_training_phase_header(
		mode='accumulative training (progressive)',
		epochs=epochs_per_round,
		num_classes=len(class_names),
		num_train_checkpoints=max_train_round,
		num_test_checkpoints=len(test_ckps),
		total_samples=sum(len(train_data[c]) for c in train_ckps),
	)

	final_model = None
	lr = config.get('training', {}).get('learning_rate', 1e-4)
	wd = config.get('training', {}).get('weight_decay', 1e-4)

	for round_idx in range(1, max_train_round + 1):
		curr_train_ckp = f"ckp_{round_idx}"
		next_test_ckp = f"ckp_{round_idx + 1}"
		logger.info(f"\nRound {round_idx} - Training up to {curr_train_ckp}, validating on {curr_train_ckp}, testing on {next_test_ckp}")

		# fresh model each round
		model = create_model(config).to(device)

		# dataloaders for accumulative round
		train_loader, val_loader, _ = get_dataloaders(config, mode='accumulative', current_checkpoint=curr_train_ckp)


		criterion = nn.CrossEntropyLoss()
		optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

		use_cuda = torch.cuda.is_available()

		# Prepare next-checkpoint test loader once per round
		from .common import _build_simple_dataset
		from torch.utils.data import DataLoader
		next_samples = test_data.get(next_test_ckp, [])
		next_test_loader = None
		if next_samples:
			class_to_idx = {n: i for i, n in enumerate(class_names)}
			ds_next = _build_simple_dataset(next_samples, class_to_idx)
			next_test_loader = DataLoader(
				ds_next,
				batch_size=int(config.get('training', {}).get('eval_batch_size', 8)),
				shuffle=False,
				num_workers=0,
				pin_memory=False,
			)
			# Next-ckpt loader prepared; avoid redundant sample count prints here

		# train epochs
		model.train()
		for epoch in range(epochs_per_round):
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

				# Periodic cleanup to reduce VRAM growth
				del images, labels, outputs, predicted
				if torch.cuda.is_available() and (batch_idx % 10 == 0):
					torch.cuda.empty_cache()
					if hasattr(torch.cuda, 'ipc_collect'):
						torch.cuda.ipc_collect()

			train_loss = running_loss / max(len(train_loader), 1)
			train_acc = (correct / total) if total > 0 else 0.0
			print(f"    🔹 Epoch {epoch:2d} [TRAIN] Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Bal.Acc: {train_acc:.4f}")

			if args.train_val and val_loader and len(val_loader) > 0:
				v_loss, v_acc, v_ba, v_n = evaluate_epoch(model, val_loader, criterion, device)
				print(f"    🔸 Epoch {epoch:2d} [ VAL ] Loss: {v_loss:.4f} | Acc: {v_acc:.4f} | Bal.Acc: {v_ba:.4f}")

			# Optional per-epoch test on NEXT checkpoint when requested
			if args.train_test and next_test_loader is not None and len(next_test_loader) > 0:
				t_loss, t_acc, t_ba, t_n = evaluate_epoch(model, next_test_loader, criterion, device)
				print(f"    🔻 Epoch {epoch:2d} [TEST+] Loss: {t_loss:.4f} | Acc: {t_acc:.4f} | Bal.Acc: {t_ba:.4f}")

		# optional validation summary when not per-epoch
		if val_loader and not args.train_val:
			v_loss, v_acc, v_ba, v_n = evaluate_epoch(model, val_loader, criterion, device)
			logger.info(f"    📊 Round {round_idx} Validation → {curr_train_ckp}: Acc: {v_acc*100:.2f}% | Bal.Acc: {v_ba*100:.2f}%")

		# End-of-round test summary if not tested per-epoch
		if (not args.train_test) and next_test_loader is not None and len(next_test_loader) > 0:
			t_loss, t_acc, t_ba, t_n = evaluate_epoch(model, next_test_loader, criterion, device)
			logger.info(f"    📊 Round {round_idx} Test → {next_test_ckp}: Acc: {t_acc*100:.2f}% | Bal.Acc: {t_ba*100:.2f}% ({t_n} samples)")

		# save final round model
		if round_idx == max_train_round:
			final_model = model
			try:
				torch.save(model.state_dict(), 'accumulative_model.pth')
				logger.info("💾 Final model saved to accumulative_model.pth")
			except Exception as e:
				logger.warning(f"Failed to save accumulative_model.pth: {e}")
		else:
			# clean
			del model, optimizer, criterion
			if torch.cuda.is_available():
				torch.cuda.empty_cache()

	icicle_logger.log_training_completion(f"accumulative ({max_train_round} rounds)")
	return final_model

