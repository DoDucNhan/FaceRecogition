import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from model.facenet_lightning import FaceNetLightning
from data.data_module import FaceDataModule
from utils.checkpoint_callback import RobustCheckpointCallback
from utils.checkpoint_utils import find_valid_checkpoint, load_training_state, load_metrics_for_callback
from utils.metrics_callback import MetricsLogger

torch.set_float32_matmul_precision('medium')
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="FaceNet Finetuning with PyTorch Lightning")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--state_dir', type=str, default='./states', 
                        help='Directory for the training and metrics state')
    parser.add_argument('--min_images', type=int, default=2, help='Minimum images per person')
    parser.add_argument('--validate_images', action='store_true', help='Validate images before training')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, default=10, 
                        help='Patience for early stopping (epochs with no improvement before stopping, 0 to disable)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--grad_accumulation', type=int, default=2, 
                        help='Number of steps to accumulate gradients')
    
    # Arguments for ArcFace+UNPG
    parser.add_argument('--use_unpg', action='store_true', 
                        help='Use Unified Negative Pair Generation with ArcFace')
    parser.add_argument('--arcface_scale', type=float, default=64.0, 
                        help='Scale factor for ArcFace loss')
    parser.add_argument('--arcface_margin', type=float, default=0.5, 
                        help='Angular margin for ArcFace loss')
    parser.add_argument('--wisker_size', type=float, default=1.5, 
                        help='Wisker size for UNPG filtering')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training to save memory')
    
    # Freezing parameters
    parser.add_argument('--freeze_groups', type=int, default=7, help='Initial number of groups to freeze')
    parser.add_argument('--unfreeze_epoch', type=int, default=5, help='Unfreeze a group every N epochs')
    
    # Learning rate parameters
    parser.add_argument('--lr_scheduler', choices=['cosine', 'plateau', 'sgdr'], default='cosine',
                        help='Learning rate scheduler')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    
    # Checkpoint parameters
    parser.add_argument('--resume', action='store_true', help='Attempt to resume from latest checkpoint')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help='Resume from specific checkpoint path')
    parser.add_argument('--save_freq', type=int, default=1, 
                        help='Save checkpoint frequency (epochs)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads for dataloader')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default="facenet-lightning", help='Name for this experiment')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Handle resumption - first check if we should auto-detect checkpoint
    checkpoint_path = None
    start_epoch = 0
    resume_state = None
    
    if args.resume:
        # Look for valid checkpoint
        checkpoint_path = find_valid_checkpoint(args.output_dir)
        if checkpoint_path:
            print(f"Found checkpoint to resume from: {checkpoint_path}")
            # Load training state if available
            resume_state = load_training_state(args.state_dir)
            if resume_state is None:
                print(f"WARNING: Could not load previous training state from checkpoint {args.state_dir}")


    elif args.resume_from_checkpoint is not None:
        # Use explicitly provided checkpoint path
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
            print(f"Using specified checkpoint: {checkpoint_path}")
            # Load training state if available
            resume_state = load_training_state(args.state_dir)
            if resume_state is None:
                print(f"WARNING: Could not load previous training state from checkpoint {args.state_dir}")
            
    if resume_state:
        print("Loading previous state................")
        excluded_fields = [
            "output_dir", 
            "log_dir", 
            "experiment_name", 
            "resume", 
            "resume_from_checkpoint"
        ]

        # Update args directly, skipping excluded fields
        for key, value in resume_state["args"].items():
            if key not in excluded_fields:
                setattr(args, key, value)

        print(f"Resuming from epoch {resume_state['epoch']}")
        start_epoch = resume_state['epoch']
   

    # Initialize data module
    data_module = FaceDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        min_images_per_person=args.min_images,
        validate_images=args.validate_images,
        val_split=args.val_split,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Make sure data is prepared before creating model
    data_module.setup()
    
    # Create FaceNet model
    model = FaceNetLightning(
        num_classes=data_module.get_num_classes(),
        pretrained=args.pretrained,
        learning_rate=args.lr,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        use_unpg=args.use_unpg,
        arcface_scale=args.arcface_scale,
        arcface_margin=args.arcface_margin,
        wisker_size=args.wisker_size,
        freeze_initial_layers=args.freeze_groups,
        unfreeze_epoch_freq=args.unfreeze_epoch,
        weight_decay=args.weight_decay,
        mixed_precision=args.mixed_precision,
        resume_from_checkpoint=checkpoint_path
    )
    
    # Set up callbacks
    callbacks = []
    
    # Custom robust checkpoint callback
    checkpoint_callback = RobustCheckpointCallback(
        dirpath=args.output_dir,
        state_dir=args.state_dir,
        filename="facenet-{epoch:02d}-{val_accuracy:.4f}",
        monitor="val_accuracy",
        mode="max",
        save_top_k=3,
        save_last=True,
        every_n_epochs=args.save_freq,
        backup_best=True,
        args=args
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback (if enabled)
    if args.early_stopping > 0:
        early_stopping_callback = EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            patience=args.early_stopping,
            verbose=True
        )
        callbacks.append(early_stopping_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Add our custom metrics logger
    metrics_logger = MetricsLogger(
        log_dir=args.log_dir,
        state_dir=args.state_dir,
        experiment_name=args.experiment_name
    )
    if checkpoint_path:
        if load_metrics_for_callback(metrics_logger, args.state_dir):
            print("Loaded metrics from previous training run")
        else:
            print("WARNING: Metrics could not be loaded from checkpoint \n Use the default metrics logger")

    callbacks.append(metrics_logger)
    
    # Initialize Trainer
    precision = '16-mixed' if args.mixed_precision else 32
    
    # Set max epochs based on resume state if available
    max_epochs = args.epochs
    if start_epoch > 0:
        max_epochs = max(args.epochs, start_epoch + 20)  # At least 20 more epochs
    
    # Instead of passing resume_from_checkpoint to the Trainer constructor:
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        accumulate_grad_batches=args.grad_accumulation,
        precision=precision,
        log_every_n_steps=10,
        default_root_dir=args.output_dir,
        deterministic=True,  # For reproducibility
        num_sanity_val_steps=0
    )

    # If we have a checkpoint, use this pattern instead:
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        trainer.fit_loop.epoch_progress.current.processed = start_epoch
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        print("Starting training from scratch")
        trainer.fit(model, data_module)
    
    # Print best model path
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
        print(f"Best model path: {checkpoint_callback.best_model_path}")
        if hasattr(checkpoint_callback, 'best_model_score'):
            print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.ckpt')
    trainer.save_checkpoint(final_path)
    print(f"Final model saved to {final_path}")
    print("Training completed!")


if __name__ == "__main__":
    main()