import os
import sys
import argparse
import torch
import glob
from pathlib import Path
import shutil
import json


def list_checkpoints(checkpoint_dir, pattern="*.ckpt"):
    """List all checkpoints in a directory"""
    checkpoints = glob.glob(os.path.join(checkpoint_dir, pattern))
    return sorted(checkpoints, key=lambda x: os.path.getmtime(x))


def verify_checkpoint(checkpoint_path):
    """Verify a checkpoint file and print its contents"""
    try:
        print(f"Verifying checkpoint: {checkpoint_path}")
        
        # Try to load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Get basic info
        keys = list(checkpoint.keys())
        print(f"Checkpoint keys: {keys}")
        
        if "state_dict" in checkpoint:
            model_keys = list(checkpoint["state_dict"].keys())
            print(f"Model has {len(model_keys)} parameter tensors")
            
        if "hyper_parameters" in checkpoint:
            print(f"Hyperparameters: {checkpoint['hyper_parameters']}")
            
        if "optimizer_states" in checkpoint:
            print(f"Optimizer states: {len(checkpoint['optimizer_states'])}")
        
        print(f"Checkpoint is valid.")
        return True
    except Exception as e:
        print(f"Error verifying checkpoint: {str(e)}")
        return False


def find_latest_valid_checkpoint(checkpoint_dir, pattern="*.ckpt"):
    """Find the latest valid checkpoint in a directory"""
    checkpoints = list_checkpoints(checkpoint_dir, pattern)
    
    for checkpoint_path in reversed(checkpoints):
        if verify_checkpoint(checkpoint_path):
            return checkpoint_path
    
    return None


def repair_checkpoint(corrupted_path, valid_path, output_path=None):
    """Attempt to repair a corrupted checkpoint using a valid one"""
    if not output_path:
        output_path = f"{corrupted_path}.repaired"
    
    try:
        print(f"Loading valid checkpoint: {valid_path}")
        valid_checkpoint = torch.load(valid_path, map_location="cpu")
        
        try:
            print(f"Loading corrupted checkpoint: {corrupted_path}")
            corrupted_checkpoint = torch.load(corrupted_path, map_location="cpu")
            
            # Merge the checkpoints, preferring the corrupted one for state_dict
            # but using the valid one as fallback
            merged = valid_checkpoint.copy()
            
            # Try to preserve the state dict from corrupted checkpoint
            if "state_dict" in corrupted_checkpoint:
                merged["state_dict"] = corrupted_checkpoint["state_dict"]
            
            # Save the merged checkpoint
            torch.save(merged, output_path)
            print(f"Repaired checkpoint saved to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Could not load corrupted checkpoint, using valid checkpoint as base")
            print(f"Error: {str(e)}")
            
            # Just copy the valid checkpoint
            shutil.copy2(valid_path, output_path)
            print(f"Valid checkpoint copied to: {output_path}")
            return True
            
    except Exception as e:
        print(f"Failed to repair checkpoint: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Checkpoint repair utility")
    
    parser.add_argument('--dir', type=str, default='./output',
                        help='Directory containing checkpoints')
    parser.add_argument('--action', choices=['list', 'verify', 'repair', 'find'],
                        default='list', help='Action to perform')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint path to verify or repair')
    parser.add_argument('--pattern', type=str, default='*.ckpt',
                        help='Pattern to match checkpoint files')
    parser.add_argument('--valid-checkpoint', type=str,
                        help='Known valid checkpoint to use for repair')
    parser.add_argument('--output', type=str, help='Output path for repaired checkpoint')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        checkpoints = list_checkpoints(args.dir, args.pattern)
        for i, ckpt in enumerate(checkpoints):
            mtime = os.path.getmtime(ckpt)
            size = os.path.getsize(ckpt) / (1024 * 1024)  # Size in MB
            print(f"{i+1}. {ckpt} - Modified: {mtime}, Size: {size:.2f} MB")
    
    elif args.action == 'verify':
        if args.checkpoint:
            verify_checkpoint(args.checkpoint)
        else:
            checkpoints = list_checkpoints(args.dir, args.pattern)
            for ckpt in checkpoints:
                valid = verify_checkpoint(ckpt)
                print(f"{ckpt}: {'VALID' if valid else 'CORRUPTED'}")
                print("-" * 80)
    
    elif args.action == 'find':
        latest_valid = find_latest_valid_checkpoint(args.dir, args.pattern)
        if latest_valid:
            print(f"Latest valid checkpoint: {latest_valid}")
        else:
            print("No valid checkpoints found")
    
    elif args.action == 'repair':
        if not args.checkpoint:
            print("Error: --checkpoint is required for repair action")
            return
            
        if not args.valid_checkpoint:
            print("Finding latest valid checkpoint...")
            valid_checkpoint = find_latest_valid_checkpoint(args.dir, args.pattern)
            if not valid_checkpoint:
                print("Error: No valid checkpoint found for repair")
                return
        else:
            valid_checkpoint = args.valid_checkpoint
            
        repair_checkpoint(args.checkpoint, valid_checkpoint, args.output)


if __name__ == "__main__":
    main()