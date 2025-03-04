import os
import torch
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MetricsLogger(Callback):
    """
    Callback for tracking and visualizing training metrics similar to the original TrainingLogger
    """
    def __init__(self, log_dir, experiment_name=None):
        """
        Initialize metrics logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Optional name for this run
        """
        super().__init__()
        self.log_dir = log_dir
        
        # Create experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"experiment_{timestamp}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'learning_rate': [],
            'frozen_groups': [],
            'unfrozen_groups': []
        }
        
        # Track epochs
        self.epochs = []
        self.tabulate_file = os.path.join(self.experiment_dir, 'training_log.txt')
        
        print(f"MetricsLogger initialized. Logs will be saved to {self.experiment_dir}")
    

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of the training epoch"""
        self._log_metrics(trainer)
        

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of the validation epoch"""
        self._log_metrics(trainer)
    

    def _log_metrics(self, trainer):
        """Log metrics from current trainer state"""
        epoch = trainer.current_epoch + 1  # Convert 0-based to 1-based for logging
        
        # Only log once per epoch (after validation)
        if epoch in self.epochs:
            return
                
        self.epochs.append(epoch)
        
        # Get metrics from trainer's callback_metrics
        metrics = trainer.callback_metrics
        
        # Debug output
        # print(f"Available metrics for epoch {epoch}: {list(metrics.keys())}")
        
        # Initialize all metrics for this epoch with None first
        # This ensures all arrays have the same length
        for key in self.metrics:
            self.metrics[key].append(None)
        
        # Then update with actual values if available
        if 'train_loss' in metrics:
            train_loss = metrics['train_loss']
            if isinstance(train_loss, torch.Tensor):
                train_loss = train_loss.item()
            self.metrics['train_loss'][-1] = train_loss
        
        if 'val_accuracy' in metrics:
            val_acc = metrics['val_accuracy']
            if isinstance(val_acc, torch.Tensor):
                val_acc = val_acc.item()
            self.metrics['val_accuracy'][-1] = val_acc
        
        # Get learning rate from optimizer
        if trainer.optimizers:
            self.metrics['learning_rate'][-1] = trainer.optimizers[0].param_groups[0]['lr']
        
        # Get frozen/unfrozen groups from model
        if hasattr(trainer.lightning_module, 'current_frozen_groups'):
            self.metrics['frozen_groups'][-1] = trainer.lightning_module.current_frozen_groups
            
            # Count total groups
            if hasattr(trainer.lightning_module, '_get_layer_groups'):
                total_groups = len(trainer.lightning_module._get_layer_groups())
                self.metrics['unfrozen_groups'][-1] = total_groups - trainer.lightning_module.current_frozen_groups
        
        # Save and visualize metrics
        self._save_metrics()
        self._save_tabulated_text()
        self.plot_metrics()


    def _save_metrics(self):
        """Save metrics to JSON file"""
        metrics_data = {
            'epochs': self.epochs,
            **self.metrics
        }
        
        metrics_file = os.path.join(self.experiment_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=4)

    def _save_tabulated_text(self):
        """Save metrics to text file using tabulate library for pretty formatting"""
        # Get current data
        df = self.get_dataframe()
        
        # Format column names: snake_case to Title Case
        formatted_columns = {col: ' '.join(word.capitalize() for word in col.split('_')) 
                           for col in df.columns}
        df = df.rename(columns=formatted_columns)
        
        # Generate tabulated table with current data
        table = tabulate(
            df, 
            headers='keys',
            tablefmt='fancy_grid',   # Use fancy_grid for nice borders
            floatfmt='.7f',          # Format floats to 7 decimal places
            numalign='right',        # Right-align numbers
            showindex=False          # Don't show row indices
        )
        
        # Write or overwrite the file with the full table
        with open(self.tabulate_file, 'w') as tabfile:
            tabfile.write(f"Training Log: {self.experiment_name}\n")
            tabfile.write("=" * 80 + "\n\n")
            tabfile.write(table)
            tabfile.write("\n\nGenerated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    

    def get_dataframe(self):
        """
        Get metrics as a pandas DataFrame with safety checks
        
        Returns:
            pd.DataFrame: DataFrame containing all logged metrics
        """
        # First ensure all arrays have the same length
        array_lengths = {key: len(value) for key, value in self.metrics.items()}
        unique_lengths = set(array_lengths.values())
        epochs_length = len(self.epochs)
        
        if len(unique_lengths) > 1 or (unique_lengths and next(iter(unique_lengths)) != epochs_length):
            print("Warning: Metric arrays have inconsistent lengths. Fixing...")
            # Find the max length
            max_length = max([epochs_length] + list(array_lengths.values()))
            
            # Extend arrays that are too short
            if len(self.epochs) < max_length:
                self.epochs.extend([None] * (max_length - len(self.epochs)))
                
            for key, value in self.metrics.items():
                if len(value) < max_length:
                    value.extend([None] * (max_length - len(value)))
        
        data = {'epoch': self.epochs}
        data.update(self.metrics)
        return pd.DataFrame(data)


    def plot_metrics(self):
        """Plot and save training metrics, ignoring rows with None values"""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        # Filter out None values for each metric and plot only valid data
        
        # Plot training loss
        valid_loss_data = [(epoch, loss) for epoch, loss in zip(self.epochs, self.metrics['train_loss']) if loss is not None]
        if valid_loss_data:
            valid_epochs, valid_losses = zip(*valid_loss_data)
            axes[0, 0].plot(valid_epochs, valid_losses, 'b-', marker='o')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        else:
            axes[0, 0].text(0.5, 0.5, 'No training loss data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0, 0].transAxes)
        
        # Plot validation accuracy
        valid_acc_data = [(epoch, acc) for epoch, acc in zip(self.epochs, self.metrics['val_accuracy']) if acc is not None]
        if valid_acc_data:
            valid_epochs, valid_accs = zip(*valid_acc_data)
            axes[0, 1].plot(valid_epochs, valid_accs, 'g-', marker='o')
            axes[0, 1].set_title('Validation Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].grid(True)
        else:
            axes[0, 1].text(0.5, 0.5, 'No validation accuracy data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[0, 1].transAxes)
        
        # Plot learning rate
        valid_lr_data = [(epoch, lr) for epoch, lr in zip(self.epochs, self.metrics['learning_rate']) if lr is not None]
        if valid_lr_data:
            valid_epochs, valid_lrs = zip(*valid_lr_data)
            axes[1, 0].plot(valid_epochs, valid_lrs, 'c-', marker='o')
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')  # Log scale for learning rate
            axes[1, 0].grid(True)
        else:
            axes[1, 0].text(0.5, 0.5, 'No learning rate data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 0].transAxes)
        
        # Plot unfrozen groups
        valid_ug_data = [(epoch, ug) for epoch, ug in zip(self.epochs, self.metrics['unfrozen_groups']) if ug is not None]
        if valid_ug_data:
            valid_epochs, valid_ugs = zip(*valid_ug_data)
            axes[1, 1].plot(valid_epochs, valid_ugs, 'r-', marker='o')
            axes[1, 1].set_title('Unfrozen Layer Groups')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Number of Unfrozen Groups')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'No unfrozen groups data available', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=axes[1, 1].transAxes)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
        plt.savefig(os.path.join(self.experiment_dir, 'training_metrics.png'), dpi=200)
        plt.close()


    def load_from_file(self, metrics_file):
        """
        Load metrics from a JSON file
        
        Args:
            metrics_file: Path to metrics JSON file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(metrics_file):
            print(f"Metrics file {metrics_file} does not exist")
            return False
        
        try:
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            
            self.epochs = metrics_data['epochs']
            for key in self.metrics:
                if key in metrics_data:
                    self.metrics[key] = metrics_data[key]
            
            print(f"Loaded metrics from {metrics_file}")
            return True
        except Exception as e:
            print(f"Failed to load metrics: {str(e)}")
            return False