import os
import torch
import json
import glob
from pathlib import Path


def find_latest_checkpoint(checkpoint_dir, pattern="facenet-*.ckpt"):
    """
    Find the latest checkpoint in a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern to match checkpoint files
        
    Returns:
        path: Path to the latest checkpoint, or None if not found
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoints:
        # Also check for Lightning's last.ckpt file
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt):
            return last_ckpt
        return None
    
    # Sort by modification time (newest first)
    checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)
    return checkpoints[0]


def save_training_state(args, epoch, output_dir, filename="training_state.json"):
    """
    Save training arguments and state for later resumption
    
    Args:
        args: Training arguments
        epoch: Current epoch
        output_dir: Directory to save state
        filename: Name of the state file
    """
    state = {
        "args": vars(args),
        "epoch": epoch,
    }
    
    state_path = os.path.join(output_dir, filename)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    
    print(f"Saved training state to {state_path}")


def load_training_state(output_dir, filename="training_state.json"):
    """
    Load training arguments and state for resumption
    
    Args:
        output_dir: Directory containing state file
        filename: Name of the state file
        
    Returns:
        state: Dictionary containing training state, or None if not found
    """
    state_path = os.path.join(output_dir, filename)
    if not os.path.exists(state_path):
        return None
    
    with open(state_path, "r") as f:
        state = json.load(f)
    
    print(f"Loaded training state from {state_path}")
    return state


def verify_checkpoint_integrity(checkpoint_path):
    """
    Verify that a checkpoint file is not corrupted
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        bool: True if checkpoint is valid, False otherwise
    """
    try:
        # Try to load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Check if essential keys are present
        required_keys = ["state_dict", "hyper_parameters"]
        if not all(key in checkpoint for key in required_keys):
            print(f"Warning: Checkpoint at {checkpoint_path} is missing required keys")
            return False
            
        print(f"Checkpoint at {checkpoint_path} verified successfully")
        return True
    except Exception as e:
        print(f"Error verifying checkpoint at {checkpoint_path}: {str(e)}")
        return False


def create_backup_checkpoint(checkpoint_path):
    """
    Create a backup of a checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        backup_path: Path to backup file
    """
    import shutil
    
    checkpoint_path = Path(checkpoint_path)
    backup_path = checkpoint_path.parent / f"{checkpoint_path.stem}_backup{checkpoint_path.suffix}"
    
    try:
        shutil.copy2(checkpoint_path, backup_path)
        print(f"Created backup of checkpoint at {backup_path}")
        return str(backup_path)
    except Exception as e:
        print(f"Error creating backup of checkpoint: {str(e)}")
        return None


def find_valid_checkpoint(checkpoint_dir, pattern="facenet-*.ckpt", max_attempts=3):
    """
    Find the most recent valid checkpoint in a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        pattern: Glob pattern to match checkpoint files
        max_attempts: Maximum number of checkpoints to try
        
    Returns:
        path: Path to the most recent valid checkpoint, or None if not found
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    
    if not checkpoints:
        # Also check for Lightning's last.ckpt file
        last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
        if os.path.exists(last_ckpt) and verify_checkpoint_integrity(last_ckpt):
            return last_ckpt
        return None
    
    # Sort by modification time (newest first)
    checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Try checkpoints one by one until we find a valid one
    for i, ckpt in enumerate(checkpoints[:max_attempts]):
        if verify_checkpoint_integrity(ckpt):
            return ckpt
    
    return None


def load_metrics_for_callback(metrics_callback, checkpoint_path):
    """
    Load metrics from checkpoint directory for callback
    
    Args:
        metrics_callback: MetricsLogger instance
        checkpoint_path: Path to checkpoint file
        
    Returns:
        bool: True if metrics were loaded successfully
    """
    # Try to find metrics.json in the same directory as the checkpoint
    checkpoint_dir = os.path.dirname(checkpoint_path)
    metrics_file = os.path.join(checkpoint_dir, 'metrics.json')
    
    if os.path.exists(metrics_file):
        return metrics_callback.load_from_file(metrics_file)
    
    # If not found, try to find it in the parent directory
    metrics_file = os.path.join(os.path.dirname(checkpoint_dir), 'metrics.json')
    if os.path.exists(metrics_file):
        return metrics_callback.load_from_file(metrics_file)
    
    print("No metrics file found for checkpoint")
    return False