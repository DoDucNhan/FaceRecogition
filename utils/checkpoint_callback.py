import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import shutil
from pathlib import Path
from .checkpoint_utils import verify_checkpoint_integrity, create_backup_checkpoint, save_training_state


import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
import shutil
from pathlib import Path
from .checkpoint_utils import verify_checkpoint_integrity, create_backup_checkpoint, save_training_state

class RobustCheckpointCallback(Callback):
    """Custom callback for robust checkpoint handling with backup and integrity check"""
    
    def __init__(self, 
                 dirpath,
                 state_dir,
                 filename="{epoch:02d}-{val_accuracy:.4f}",
                 monitor="val_accuracy",
                 mode="max",
                 save_top_k=3,
                 save_last=True,
                 every_n_epochs=1,
                 backup_best=True,
                 args=None,
                 save_pt=True):
        """
        Initialize robust checkpoint callback
        
        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename format (without extension)
            monitor: Metric to monitor
            mode: 'min' or 'max' for the monitored metric
            save_top_k: Number of best models to save
            save_last: Whether to save the last model
            every_n_epochs: Save checkpoints every N epochs
            backup_best: Whether to create backups of best models
            args: Training arguments for state saving
            save_pt: Whether to save .pt model files alongside checkpoints
        """
        super().__init__()
        self.dirpath = dirpath
        self.state_dir = state_dir
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.backup_best = backup_best
        self.args = args
        self.save_pt = save_pt
        
        # Tracked for best models
        self.best_k_models = {}
        self.kth_best_model_path = ""
        self.best_model_score = torch.tensor(float('inf') if mode == "min" else float('-inf'))
        self.best_model_path = ""
        
        # Create directory if it doesn't exist
        os.makedirs(dirpath, exist_ok=True)
    
    
    def _save_model(self, trainer, pl_module, epoch, logs=None):
        """Save a model checkpoint"""
        logs = logs or {}
        monitor_val = logs.get(self.monitor, None)
        
        # Get filepath
        filepath = os.path.join(
            self.dirpath, 
            self.filename.format(epoch=epoch, **logs) + ".ckpt"
        )
        
        # Save checkpoint
        trainer.save_checkpoint(filepath)
        print(f"Saved checkpoint to {filepath}")
        
        # Also save as .pt format if requested
        if self.save_pt:
            pt_filepath = os.path.join(
                self.dirpath, 
                self.filename.format(epoch=epoch, **logs) + ".pt"
            )
            # Save just the model state dict
            torch.save(pl_module.model.state_dict(), pt_filepath)
            print(f"Saved model state dict to {pt_filepath}")
        
        # Verify checkpoint integrity
        if verify_checkpoint_integrity(filepath):
            # Save additional state if args provided
            if self.args:
                save_training_state(self.args, epoch, self.state_dir)
            
            # Check if this is best model
            if monitor_val is not None:
                current_score = torch.tensor(monitor_val)
                if self.mode == "min":
                    better = current_score < self.best_model_score
                else:  # mode == "max"
                    better = current_score > self.best_model_score
                
                if better:
                    print(f"New best model! Score: {current_score:.6f} (previous: {self.best_model_score:.6f})")
                    
                    if self.backup_best and self.best_model_path:
                        # Create backup of previous best model
                        create_backup_checkpoint(self.best_model_path)
                    
                    # Update best model info
                    self.best_model_score = current_score
                    self.best_model_path = filepath
                    
                    # Save best model separately 
                    best_model_path = os.path.join(self.dirpath, "best_model.ckpt")
                    shutil.copy2(filepath, best_model_path)
                    print(f"Saved best model checkpoint to {best_model_path}")
                    
                    # Save best model separately in .pt format
                    if self.save_pt:
                        best_pt_path = os.path.join(self.dirpath, "best_model.pt")
                        torch.save(pl_module.model.state_dict(), best_pt_path)
                        print(f"Saved best model state dict to {best_pt_path}")
                    
                    # Update best k models dict
                    self.best_k_models[filepath] = current_score
                    
                    # If we have more than save_top_k models, remove the worst
                    if len(self.best_k_models) > self.save_top_k > 0:
                        if self.mode == "min":
                            worst_score = max(self.best_k_models.values())
                            worst_paths = [
                                k for k, v in self.best_k_models.items() 
                                if v == worst_score
                            ]
                        else:
                            worst_score = min(self.best_k_models.values())
                            worst_paths = [
                                k for k, v in self.best_k_models.items() 
                                if v == worst_score
                            ]
                        
                        # Get the oldest checkpoint with the worst score
                        worst_path = sorted(worst_paths, key=os.path.getctime)[0]
                        
                        # Remove worst model
                        if os.path.exists(worst_path) and worst_path != filepath:
                            os.remove(worst_path)
                            print(f"Removed old checkpoint: {worst_path}")
                            
                            # Also remove corresponding .pt file if it exists
                            pt_worst_path = worst_path.replace('.ckpt', '.pt')
                            if os.path.exists(pt_worst_path):
                                os.remove(pt_worst_path)
                        
                        # Remove from dict
                        self.best_k_models.pop(worst_path)
            
            return filepath
        else:
            print(f"Warning: Checkpoint at {filepath} failed integrity check")
            return None


    def on_train_epoch_end(self, trainer, pl_module):
        """Called when the train epoch ends"""
        epoch = trainer.current_epoch
        
        # ALWAYS save latest checkpoint - this gets overwritten each epoch
        latest_filepath = os.path.join(self.dirpath, "latest.ckpt")
        trainer.save_checkpoint(latest_filepath)
        print(f"Saved latest checkpoint to {latest_filepath}")
        
        # ALWAYS save latest model state dict
        if self.save_pt:
            latest_pt_path = os.path.join(self.dirpath, "latest_model.pt")
            torch.save(pl_module.model.state_dict(), latest_pt_path)
            print(f"Saved latest model state dict to {latest_pt_path}")
        
        # Get current logs for metrics
        logs = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in trainer.callback_metrics.items()}
        
        # Save regular checkpoint based on frequency if not already saved as a best model
        if (epoch + 1) % self.every_n_epochs == 0:
            filepath = self._save_model(trainer, pl_module, epoch + 1, logs)
            
            # Save last checkpoint
            if self.save_last and filepath and os.path.exists(filepath):
                last_filepath = os.path.join(self.dirpath, "last.ckpt")
                shutil.copy2(filepath, last_filepath)
                print(f"Copied checkpoint to {last_filepath}")
                
                # Also save last model in .pt format
                if self.save_pt:
                    last_pt_path = os.path.join(self.dirpath, "last_model.pt")
                    torch.save(pl_module.model.state_dict(), last_pt_path)
                    print(f"Saved last model state dict to {last_pt_path}")

        # Check if this might be a new best model
        monitor_val = logs.get(self.monitor, None)
        if monitor_val is not None:
            current_score = torch.tensor(monitor_val)
            if self.mode == "min":
                better = current_score < self.best_model_score
            else:  # mode == "max"
                better = current_score > self.best_model_score
            
            # If this is potentially a new best model, force save regardless of frequency
            if better:
                print(f"Potential new best model detected with {self.monitor}={current_score:.6f}")
                filepath = self._save_model(trainer, pl_module, epoch + 1, logs)
    

    def on_train_end(self, trainer, pl_module):
        """Called when training ends"""
        # Save final model
        final_path = os.path.join(self.dirpath, "final_model.ckpt")
        trainer.save_checkpoint(final_path)
        
        # Save final model in .pt format
        if self.save_pt:
            final_pt_path = os.path.join(self.dirpath, "final_model.pt")
            torch.save(pl_module.model.state_dict(), final_pt_path)
            print(f"Saved final model state dict to {final_pt_path}")
        
        # Create symlink to best model as best_model.ckpt
        if self.best_model_path and os.path.exists(self.best_model_path):
            best_symlink = os.path.join(self.dirpath, "best_model.ckpt")
            
            # On Windows, may need to remove existing symlink first
            if os.path.exists(best_symlink):
                os.remove(best_symlink)
            
            # Create symlink or copy if symlink not supported
            try:
                os.symlink(self.best_model_path, best_symlink)
            except (OSError, NotImplementedError):
                shutil.copy2(self.best_model_path, best_symlink)