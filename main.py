import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from model.facenet_lightning import FaceNetLightning
from data.data_module import FaceDataModule
from utils.helpers import set_seed
from utils.checkpoint_callback import RobustCheckpointCallback
from utils.checkpoint_utils import find_valid_checkpoint, load_training_state
from utils.metrics_callback import MetricsLogger

# torch.set_float32_matmul_precision('medium' | 'high')
torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="FaceNet Finetuning with PyTorch Lightning")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--min_images', type=int, default=2, help='Minimum images per person')
    parser.add_argument('--validate_images', action='store_true', help='Validate images before training')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, default=10, 
                        help='Patience for early stopping (epochs with no improvement before stopping, 0 to disable)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    
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
    
    # Set random seed
    set_seed(args.seed)
    
    # Handle resumption - first check if we should auto-detect checkpoint
    checkpoint_path = None
    start_epoch = 0
    resume_state = None
    
    print(f"Resume flag: {args.resume}")
    print(f"Resuming from specific checkpoint: {args.resume_from_checkpoint}")
    if args.resume:
        # Look for valid checkpoint
        checkpoint_path = find_valid_checkpoint(args.output_dir)
        if checkpoint_path:
            print(f"Found checkpoint to resume from: {checkpoint_path}")
            # Load training state if available
            resume_state = load_training_state(args.output_dir)
            if resume_state:
                print(f"Resuming from epoch {resume_state['epoch']}")
                start_epoch = resume_state['epoch']

    elif args.resume_from_checkpoint is not None:
        # Use explicitly provided checkpoint path
        if os.path.exists(args.resume_from_checkpoint):
            checkpoint_path = args.resume_from_checkpoint
            print(f"Using specified checkpoint: {checkpoint_path}")
            # Load training state if available
            resume_state = load_training_state(args.output_dir)
            if resume_state:
                print(f"Resuming from epoch {resume_state['epoch']}")
                start_epoch = resume_state['epoch']
    print("Checkpoint path:", checkpoint_path)
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
    
    # Set up logger
    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.experiment_name,
        version=None  # Auto-incremented version
    )

    # Add our custom metrics logger
    metrics_logger = MetricsLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name
    )
    callbacks.append(metrics_logger)
    
    # Initialize Trainer
    precision = '16-mixed' if args.mixed_precision else 32
    
    # Set max epochs based on resume state if available
    max_epochs = args.epochs
    if start_epoch > 0:
        max_epochs = max(args.epochs, start_epoch + 5)  # At least 5 more epochs
    
    # Instead of passing resume_from_checkpoint to the Trainer constructor:
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        log_every_n_steps=10,
        default_root_dir=args.output_dir,
        deterministic=True,  # For reproducibility
        num_sanity_val_steps=0
    )

    # If we have a checkpoint, use this pattern instead:
    if checkpoint_path:
        # print(f"Loading checkpoint from {checkpoint_path} but skipping optimizer state")
        # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        
        # Set current epoch from checkpoint if available
        # if 'epoch' in checkpoint:
        #     current_epoch = checkpoint['epoch']
        #     print(f"Resuming from epoch {current_epoch}")
        print(f"Loading checkpoint from {checkpoint_path}")
        # checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        # # Load the model state dict
        # model.load_state_dict(checkpoint['state_dict'], strict=False)
        # print("Model weights loaded successfully")
        # # Set model-specific variables
        # if 'current_frozen_groups' in checkpoint and hasattr(model, 'current_frozen_groups'):
        #     print(f"Setting current_frozen_groups to {checkpoint['current_frozen_groups']}")
        #     model.current_frozen_groups = checkpoint['current_frozen_groups']
        
        # if 'prev_accuracy' in checkpoint and hasattr(model, 'prev_accuracy'):
        #     model.prev_accuracy = checkpoint['prev_accuracy']
        
        # if 'adaptive_patience_counter' in checkpoint and hasattr(model, 'adaptive_patience_counter'):
        #     model.adaptive_patience_counter = checkpoint['adaptive_patience_counter']

        # from utils.checkpoint_utils import load_metrics_for_callback
        # metrics_logger = load_metrics_for_callback(metrics_logger, checkpoint_path)
        # print("Loaded metrics from previous training run")

        # Do not pass checkpoint_path to trainer.fit
        trainer.fit_loop.epoch_progress.current.processed = start_epoch - 1
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    else:
        trainer.fit(model, data_module)

    # if checkpoint_path:
    #     trainer.fit(model, data_module, ckpt_path=checkpoint_path)
    #     from utils.checkpoint_utils import load_metrics_for_callback
    #     load_metrics_for_callback(metrics_logger, checkpoint_path)
    #     print("Loaded metrics from previous training run")
    # else:
    #     trainer.fit(model, data_module)
    
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