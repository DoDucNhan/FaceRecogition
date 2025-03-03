import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from tabulate import tabulate

class TrainingLogger:
    """
    Logger for tracking and visualizing training metrics
    """
    def __init__(self, log_dir, experiment_name=None):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Optional name for this run
        """
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
            'val_far': [],
            'learning_rate': []
        }
        
        # Track epochs
        self.epochs = []
        self.tabulate_file = os.path.join(self.experiment_dir, 'training_log.txt')
        self.tabulate_header_written = False
        
        print(f"Logger initialized. Logs will be saved to {self.experiment_dir}")
    
    def log_epoch(self, epoch, train_loss, val_accuracy, val_far, learning_rate, extra_metrics=None):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_accuracy: Validation accuracy
            val_far: False accept rate
            learning_rate: Current learning rate
            extra_metrics: Optional dictionary of additional metrics to log
        """
        self.epochs.append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['val_accuracy'].append(val_accuracy)
        self.metrics['val_far'].append(val_far)
        self.metrics['learning_rate'].append(learning_rate)
        
        # Add any extra metrics
        if extra_metrics:
            for key, value in extra_metrics.items():
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
        
        # Save metrics to various formats
        self._save_metrics()
        self._save_tabulated_text()
        
        # Plot and save figures
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
            floatfmt='.4f',          # Format floats to 4 decimal places
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
        Get metrics as a pandas DataFrame
        
        Returns:
            pd.DataFrame: DataFrame containing all logged metrics
        """
        data = {'epoch': self.epochs}
        data.update(self.metrics)
        return pd.DataFrame(data)


    def plot_metrics(self):
        """Plot and save training metrics"""
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        # Plot training loss
        axes[0, 0].plot(self.epochs, self.metrics['train_loss'], 'b-', marker='o')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Plot validation accuracy
        axes[0, 1].plot(self.epochs, self.metrics['val_accuracy'], 'g-', marker='o')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True)
        
        # Plot False Accept Rate (FAR)
        axes[1, 0].plot(self.epochs, self.metrics['val_far'], 'r-', marker='o')
        axes[1, 0].set_title('False Accept Rate (FAR)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('FAR')
        axes[1, 0].grid(True)
        
        # Plot learning rate
        axes[1, 1].plot(self.epochs, self.metrics['learning_rate'], 'c-', marker='o')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')  # Log scale for learning rate
        axes[1, 1].grid(True)
        
        # Adjust layout and save
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
        plt.savefig(os.path.join(self.experiment_dir, 'training_metrics.png'), dpi=200)
        plt.close()

    def load_from_file(self, metrics_file):
        """
        Load metrics from a JSON file
        
        Args:
            metrics_file: Path to metrics JSON file
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