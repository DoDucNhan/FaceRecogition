import torch
import os

def save_checkpoint(model, optimizer, scheduler, epoch, best_accuracy, 
                    label_to_idx, frozen_groups, args, save_path):
    """
    Save training checkpoint for resumption
    
    Args:
        model: FaceNet model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_accuracy: Best validation accuracy so far
        label_to_idx: Mapping from person IDs to numeric labels
        frozen_groups: Number of currently frozen layer groups
        args: Training arguments
        save_path: Path to save checkpoint
    """
    # Additional ArcFace state
    arcface_state = None
    if args.use_arcface and hasattr(model, 'criterion'):
        arcface_state = model.criterion.state_dict()

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'best_accuracy': best_accuracy,
        'label_to_idx': label_to_idx,
        'frozen_groups': frozen_groups,
        'arcface_state': arcface_state,
        'args': args.__dict__ if hasattr(args, "__dict__") else args
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

    # Copy metrics.json to the checkpoint directory if it exists
    if hasattr(args, 'log_dir') and args.log_dir:
        metrics_path = None
        if hasattr(args, 'experiment_name') and args.experiment_name:
            metrics_path = os.path.join(args.log_dir, args.experiment_name, 'metrics.json')
        else:
            # Try to find the most recent experiment directory
            log_dirs = [d for d in os.listdir(args.log_dir) 
                      if os.path.isdir(os.path.join(args.log_dir, d))]
            if log_dirs:
                # Sort by creation time (newest first)
                newest_dir = sorted(log_dirs, 
                                 key=lambda d: os.path.getctime(os.path.join(args.log_dir, d)),
                                 reverse=True)[0]
                metrics_path = os.path.join(args.log_dir, newest_dir, 'metrics.json')
        
        if metrics_path and os.path.exists(metrics_path):
            import shutil
            checkpoint_dir = os.path.dirname(save_path)
            shutil.copy(metrics_path, os.path.join(checkpoint_dir, 'metrics.json'))
            print(f"Copied metrics.json to {checkpoint_dir}")


def resume_from_checkpoint(checkpoint_path, model, optimizer, scheduler=None, criterion=None):
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: FaceNet model
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        epoch: Epoch to resume from
        best_accuracy: Best validation accuracy from checkpoint
        label_to_idx: Mapping from person IDs to numeric labels
        frozen_groups: Number of frozen layer groups
    """
    print(f"Resuming from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load ArcFace state if available and needed
    if criterion is not None and checkpoint['arcface_state'] is not None:
        criterion.load_state_dict(checkpoint['arcface_state'])
    
    # Return important states
    return (checkpoint['epoch'], checkpoint['best_accuracy'], 
            checkpoint['label_to_idx'], checkpoint['frozen_groups'])


def load_metrics_for_logger(logger, checkpoint_path):
    """
    Load metrics from checkpoint directory for logger
    
    Args:
        logger: TrainingLogger instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        bool: True if metrics were loaded successfully
    """
    # Try to find metrics.json in the same directory as the checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
    
    if os.path.exists(metrics_file):
        return logger.load_from_file(metrics_file)
    
    # If not found, try to find it in the parent directory
    metrics_file = os.path.join(os.path.dirname(checkpoint_dir), 'metrics.json')
    if os.path.exists(metrics_file):
        return logger.load_from_file(metrics_file)
    
    # If still not found, try output_dir/metrics.json
    checkpoint = torch.load(checkpoint_path)
    if hasattr(checkpoint, 'args') and hasattr(checkpoint['args'], 'output_dir'):
        metrics_file = os.path.join(checkpoint['args'].output_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            return logger.load_from_file(metrics_file)
    
    print("No metrics file found for checkpoint")
    return False