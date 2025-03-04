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
            floatfmt='.7f',          # Format floats to 4 decimal places
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
        


class TrainingLogger:
    """
    Custom logger to track and save training progress
    """
    def __init__(self, log_dir='logs', experiment_name=None):
        # Create log directory if it doesn't exist
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment name based on timestamp if not provided
        if experiment_name is None:
            experiment_name = f"experiment_{time.strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.experiment_dir / f"{experiment_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': [],
            'epochs': [],
            'stages': []
        }
        
        self.start_time = time.time()
        self.logger.info(f"Starting experiment: {experiment_name}")
        
        
    def log_hyperparams(self, params):
        """Log hyperparameters"""
        self.logger.info(f"Hyperparameters: {json.dumps(params, indent=2)}")
        
        # Save hyperparameters to file
        with open(self.experiment_dir / "hyperparams.json", 'w') as f:
            json.dump(params, f, indent=2)
    

    def log_metrics(self, metrics, step):
        """Log metrics at current step"""
        # Add metrics to history
        for key, value in metrics.items():
            if key in self.metrics:
                self.metrics[key].append(float(value))
        
        self.metrics['epochs'].append(step)
        
        # Log to console/file
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Epoch {step}: {metrics_str}")
        
        # Save metrics to file
        self.save_metrics()
    

    def log_stage(self, stage_name, step):
        """Log when a new stage begins (new layer unfrozen)"""
        self.logger.info(f"Stage change at epoch {step}: Unfreezing {stage_name}")
        self.metrics['stages'].append({
            'name': stage_name,
            'epoch': step
        })
        self.save_metrics()
    

    def save_metrics(self):
        """Save metrics to disk"""
        metrics_file = self.experiment_dir / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    

    def log_model_saved(self, path, epoch, metrics=None):
        """Log when a model is saved"""
        metrics_str = ""
        if metrics:
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        self.logger.info(f"Model saved at epoch {epoch} to {path} - {metrics_str}")
    

    def log_training_complete(self, final_metrics=None):
        """Log completion of training"""
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        self.logger.info(f"Training completed in {time_str}")
        
        if final_metrics:
            metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
            self.logger.info(f"Final metrics: {metrics_str}")
    

    def plot_metrics(self, save_path=None):
        """Plot training and validation metrics"""
        if not self.metrics['train_loss']:
            self.logger.info("No metrics to plot")
            return
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Plot loss
        plt.subplot(2, 1, 1)
        plt.plot(self.metrics['epochs'], self.metrics['train_loss'], 'b-', label='Training Loss')
        plt.plot(self.metrics['epochs'], self.metrics['val_loss'], 'r-', label='Validation Loss')
        
        # Add stage transition markers
        for stage in self.metrics['stages']:
            plt.axvline(x=stage['epoch'], color='green', linestyle='--', alpha=0.7)
            plt.text(stage['epoch'], max(self.metrics['train_loss']), 
                     f"Unfreeze {stage['name']}", rotation=90, verticalalignment='top')
        
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(2, 1, 2)
        plt.plot(self.metrics['epochs'], self.metrics['train_acc'], 'b-', label='Training Accuracy')
        plt.plot(self.metrics['epochs'], self.metrics['val_acc'], 'r-', label='Validation Accuracy')
        
        # Add stage transition markers
        for stage in self.metrics['stages']:
            plt.axvline(x=stage['epoch'], color='green', linestyle='--', alpha=0.7)
        
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            save_path = self.experiment_dir / "training_metrics.png"
        plt.savefig(save_path)
        self.logger.info(f"Training plot saved to {save_path}")
        plt.close()