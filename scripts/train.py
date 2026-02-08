#!/usr/bin/env python
"""
NeuroCircuitry Training Script

Main entry point for training connectomics segmentation models.

Usage:
    # Train with default config
    python scripts/train.py
    
    # Train with specific dataset config
    python scripts/train.py --config configs/snemi3d.yaml
    
    # Override parameters via CLI
    python scripts/train.py --config configs/snemi3d.yaml \\
        data.batch_size=8 training.max_epochs=200
    
    # Fast development run
    python scripts/train.py training.fast_dev_run=true
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ModelSummary,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from omegaconf import DictConfig, OmegaConf

# Enable Tensor Core optimization
torch.set_float32_matmul_precision("high")


def load_config(config_path: Optional[str] = None) -> DictConfig:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    # Merge CLI overrides
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_cfg)
    
    return cfg


def get_datamodule(cfg: DictConfig) -> pl.LightningDataModule:
    """Create appropriate datamodule based on config."""
    from neurocircuitry.datamodules import (
        SNEMI3DDataModule,
        CREMI3DDataModule,
        MICRONSDataModule,
    )
    
    data_cfg = cfg.data
    dataset_type = data_cfg.get("dataset", "snemi3d").lower()
    
    common_args = {
        "data_root": data_cfg.get("data_root", "data"),
        "batch_size": data_cfg.get("batch_size", 4),
        "num_workers": data_cfg.get("num_workers", 4),
        "train_val_split": data_cfg.get("train_val_split", 0.2),
        "cache_rate": data_cfg.get("cache_rate", 0.5),
        "pin_memory": data_cfg.get("pin_memory", True),
    }
    
    image_size = data_cfg.get("image_size")
    if image_size is not None:
        common_args["image_size"] = (
            tuple(image_size) if isinstance(image_size, list) else image_size
        )
    
    if dataset_type == "snemi3d":
        return SNEMI3DDataModule(
            slice_mode=data_cfg.get("slice_mode", True),
            **common_args,
        )
    
    elif dataset_type == "cremi3d":
        samples = data_cfg.get("samples")
        return CREMI3DDataModule(
            samples=list(samples) if samples else None,
            include_synapses=data_cfg.get("include_synapses", True),
            slice_mode=data_cfg.get("slice_mode", True),
            **common_args,
        )
    
    elif dataset_type == "microns":
        crop_size = data_cfg.get("crop_size")
        return MICRONSDataModule(
            volume_file=data_cfg.get("volume_file", "volume"),
            segmentation_file=data_cfg.get("segmentation_file", "segmentation"),
            include_synapses=data_cfg.get("include_synapses", False),
            include_mitochondria=data_cfg.get("include_mitochondria", False),
            slice_mode=data_cfg.get("slice_mode", True),
            crop_size=tuple(crop_size) if crop_size else None,
            **common_args,
        )
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_module(cfg: DictConfig) -> pl.LightningModule:
    """Create appropriate Lightning module based on config."""
    from neurocircuitry.modules import (
        SemanticSegmentationModule,
        InstanceSegmentationModule,
        AffinitySegmentationModule,
    )
    
    model_cfg = dict(cfg.get("model", {}))
    optimizer_cfg = dict(cfg.get("optimizer", {}))
    loss_cfg = dict(cfg.get("loss", {}))
    
    # Determine module type based on model config
    use_instance = model_cfg.pop("use_ins_head", False)
    use_affinity = model_cfg.pop("use_affinity", False)
    
    if use_affinity:
        return AffinitySegmentationModule(
            model_config=model_cfg,
            optimizer_config=optimizer_cfg,
            loss_config=loss_cfg,
        )
    elif use_instance:
        model_cfg["use_ins_head"] = True
        return InstanceSegmentationModule(
            model_config=model_cfg,
            optimizer_config=optimizer_cfg,
            loss_config=loss_cfg,
        )
    else:
        return SemanticSegmentationModule(
            model_config=model_cfg,
            optimizer_config=optimizer_cfg,
            loss_config=loss_cfg,
        )


def setup_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    """Setup training callbacks from configuration."""
    callbacks = []
    
    callback_cfg = cfg.get("callbacks", {})
    
    # Model Checkpoint
    ckpt_cfg = callback_cfg.get("checkpoint", {})
    if ckpt_cfg.get("enabled", True):
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_cfg.get("dirpath", "checkpoints"),
                filename=ckpt_cfg.get("filename", "{epoch:02d}-{val/loss:.4f}"),
                save_top_k=ckpt_cfg.get("save_top_k", 3),
                monitor=ckpt_cfg.get("monitor", "val/loss"),
                mode=ckpt_cfg.get("mode", "min"),
                save_last=ckpt_cfg.get("save_last", True),
                verbose=ckpt_cfg.get("verbose", True),
                auto_insert_metric_name=False,
            )
        )
    
    # Early Stopping
    es_cfg = callback_cfg.get("early_stopping", {})
    if es_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=es_cfg.get("monitor", "val/loss"),
                patience=es_cfg.get("patience", 20),
                mode=es_cfg.get("mode", "min"),
                verbose=es_cfg.get("verbose", True),
                min_delta=es_cfg.get("min_delta", 0.0),
            )
        )
    
    # Learning Rate Monitor
    if callback_cfg.get("lr_monitor", {}).get("enabled", True):
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    
    # Progress Bar
    callbacks.append(RichProgressBar())
    
    # Model Summary
    callbacks.append(ModelSummary(max_depth=2))
    
    return callbacks


def setup_logger(cfg: DictConfig):
    """Setup experiment logger."""
    logger_type = cfg.get("logger", "tensorboard")
    
    if logger_type == "tensorboard":
        return TensorBoardLogger(
            save_dir=cfg.get("log_dir", "logs"),
            name=cfg.get("experiment_name", "neurocircuitry"),
            version=None,
        )
    elif logger_type == "wandb":
        return WandbLogger(
            project=cfg.get("project_name", "neurocircuitry"),
            name=f"{cfg.get('experiment_name', 'run')}_{cfg.get('seed', 42)}",
            save_dir=cfg.get("log_dir", "logs"),
        )
    else:
        return True


def main():
    """Main training entry point."""
    # Parse config path from arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="NeuroCircuitry Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args, _ = parser.parse_known_args()
    
    # Load configuration
    cfg = load_config(args.config)
    
    # Print configuration
    print("=" * 60)
    print("NeuroCircuitry - Connectomics Segmentation Training")
    print("=" * 60)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed
    seed = cfg.get("seed", 42)
    pl.seed_everything(seed, workers=True)
    print(f"\nRandom seed: {seed}")
    
    # Initialize DataModule
    datamodule = get_datamodule(cfg)
    print(f"\nDataModule: {datamodule.__class__.__name__}")
    print(f"  Dataset: {cfg.data.get('dataset', 'snemi3d')}")
    print(f"  Data root: {cfg.data.get('data_root', 'data')}")
    print(f"  Batch size: {cfg.data.get('batch_size', 4)}")
    
    # Initialize Module
    module = get_module(cfg)
    print(f"\nModule: {module.__class__.__name__}")
    
    # Setup Callbacks
    callbacks = setup_callbacks(cfg)
    print(f"\nCallbacks: {len(callbacks)} registered")
    
    # Setup Logger
    logger = setup_logger(cfg)
    print(f"Logger: {cfg.get('logger', 'tensorboard')}")
    
    # Initialize Trainer
    training_cfg = cfg.training
    trainer = pl.Trainer(
        max_epochs=training_cfg.get("max_epochs", 100),
        accelerator=training_cfg.get("accelerator", "auto"),
        devices=training_cfg.get("devices", 1),
        strategy=training_cfg.get("strategy", "auto"),
        precision=training_cfg.get("precision", "32-true"),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=training_cfg.get("log_every_n_steps", 50),
        gradient_clip_val=training_cfg.get("gradient_clip_val", 1.0),
        accumulate_grad_batches=training_cfg.get("accumulate_grad_batches", 1),
        val_check_interval=training_cfg.get("val_check_interval", 1.0),
        check_val_every_n_epoch=training_cfg.get("check_val_every_n_epoch", 1),
        num_sanity_val_steps=training_cfg.get("num_sanity_val_steps", 2),
        enable_progress_bar=training_cfg.get("enable_progress_bar", True),
        enable_model_summary=training_cfg.get("enable_model_summary", True),
        deterministic=training_cfg.get("deterministic", False),
        benchmark=training_cfg.get("benchmark", True),
        fast_dev_run=training_cfg.get("fast_dev_run", False),
    )
    
    print(f"\nTrainer initialized:")
    print(f"  Max epochs: {training_cfg.get('max_epochs', 100)}")
    print(f"  Accelerator: {training_cfg.get('accelerator', 'auto')}")
    print(f"  Devices: {training_cfg.get('devices', 1)}")
    print(f"  Precision: {training_cfg.get('precision', '32-true')}")
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    
    try:
        trainer.fit(module, datamodule)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nTraining failed: {e}")
        raise
    
    # Save final checkpoint
    if trainer.global_rank == 0:
        final_path = Path("checkpoints") / "final_model.ckpt"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(final_path))
        print(f"\nFinal model saved: {final_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    
    return trainer.callback_metrics


if __name__ == "__main__":
    main()
