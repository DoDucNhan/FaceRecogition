import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from utils.logger import TrainingLogger

def visualize_metrics(metrics_file, output_file=None):
    """
    Visualize metrics from a JSON file
    
    Args:
        metrics_file: Path to metrics JSON file
        output_file: Path to save visualization (optional)
    """
    # Load metrics
    logger = TrainingLogger(os.path.dirname(metrics_file))
    if not logger.load_from_file(metrics_file):
        print(f"Failed to load metrics from {metrics_file}")
        return
    
    # Generate visualization
    logger.plot_metrics()
    
    if output_file:
        # Copy the generated plot to the specified output file
        import shutil
        plot_file = os.path.join(logger.experiment_dir, 'training_metrics.png')
        if os.path.exists(plot_file):
            shutil.copy(plot_file, output_file)
            print(f"Saved visualization to {output_file}")
    
    print("Visualization complete")

def visualize_multiple_experiments(experiment_dirs, output_file=None):
    """
    Visualize and compare metrics from multiple experiments
    
    Args:
        experiment_dirs: List of experiment directories
        output_file: Path to save visualization (optional)
    """
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Comparison of Training Runs', fontsize=16)
    
    # Colors for different experiments
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    
    # Load metrics from each experiment
    for i, exp_dir in enumerate(experiment_dirs):
        metrics_file = os.path.join(exp_dir, 'metrics.json')
        if not os.path.exists(metrics_file):
            print(f"Metrics file not found in {exp_dir}")
            continue
        
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            color = colors[i % len(colors)]
            label = os.path.basename(exp_dir)
            
            # Plot training loss
            axes[0, 0].plot(metrics['epochs'], metrics['train_loss'], 
                         f'{color}-', marker='o', label=label)
            
            # Plot validation accuracy
            axes[0, 1].plot(metrics['epochs'], metrics['val_accuracy'], 
                         f'{color}-', marker='o', label=label)
            
            # Plot FAR
            axes[1, 0].plot(metrics['epochs'], metrics['val_far'], 
                         f'{color}-', marker='o', label=label)
            
            # Plot learning rate
            axes[1, 1].plot(metrics['epochs'], metrics['learning_rate'], 
                         f'{color}-', marker='o', label=label)
            
        except Exception as e:
            print(f"Failed to process {exp_dir}: {str(e)}")
    
    # Set titles and labels
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    axes[1, 0].set_title('False Accept Rate (FAR)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('FAR')
    axes[1, 0].grid(True)
    axes[1, 0].legend()
    
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True)
    axes[1, 1].legend()
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if output_file:
        plt.savefig(output_file, dpi=200)
        print(f"Saved comparison visualization to {output_file}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument('--metrics_file', type=str, help='Path to metrics.json file')
    parser.add_argument('--experiment_dirs', type=str, nargs='+', 
                        help='Directories containing experiment metrics for comparison')
    parser.add_argument('--output', type=str, help='Path to save visualization')
    
    args = parser.parse_args()
    
    if args.metrics_file:
        visualize_metrics(args.metrics_file, args.output)
    elif args.experiment_dirs:
        visualize_multiple_experiments(args.experiment_dirs, args.output)
    else:
        parser.error("Either --metrics_file or --experiment_dirs must be provided")

if __name__ == "__main__":
    main()