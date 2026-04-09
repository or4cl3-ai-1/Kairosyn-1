#!/usr/bin/env python3
"""
KAIROSYN Phase 1: Supervised Fine-Tuning Entry Point
======================================================
Usage:
    python scripts/train_sft.py \
        --config configs/training/sft_config.yaml \
        --model_config configs/model/kairosyn_e4b.yaml \
        --output_dir ./checkpoints/kairosyn-sft-v1
"""

import argparse
from pathlib import Path

from kairosyn.model.backbone import KairosynConfig
from kairosyn.training.sft_trainer import KairosynSFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="KAIROSYN SFT Training")
    parser.add_argument(
        "--model_config", type=str,
        default="configs/model/kairosyn_e4b.yaml",
        help="Path to model config YAML"
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="./checkpoints/kairosyn-sft-v1",
        help="Output directory for checkpoints"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--no_wandb", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = KairosynConfig.from_yaml(args.model_config) \
        if Path(args.model_config).exists() \
        else KairosynConfig()

    trainer = KairosynSFTTrainer(
        config=config,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        use_wandb=not args.no_wandb,
    )
    trainer.train()


if __name__ == "__main__":
    main()
