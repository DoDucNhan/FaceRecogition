from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DemographicEvaluator:
    """Monitor performance across different demographic groups"""
    def __init__(self, model, dataloader, demographics_map, device, output_dir):
        """
        Initialize demographic evaluator
        
        Args:
            model: Face recognition model
            dataloader: DataLoader containing evaluation data
            demographics_map: Dictionary mapping label_ids to demographic groups
            device: Device to run evaluation on
            output_dir: Directory to save evaluation results
        """
        self.model = model
        self.dataloader = dataloader
        self.demographics_map = demographics_map
        self.device = device
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'demographic_eval'), exist_ok=True)
        
    def evaluate(self, epoch):
        """
        Evaluate model performance across demographic groups
        
        Args:
            epoch: Current training epoch
            
        Returns:
            results: Dictionary with evaluation metrics by demographic group
        """
        self.model.eval()
        all_embeddings = []
        all_labels = []
        
        # Collect embeddings and labels
        with torch.no_grad():
            for images, labels in self.dataloader:
                images = images.to(self.device)
                embeddings = self.model(images)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(labels)
        
        # Concatenate tensors
        embeddings = torch.cat(all_embeddings, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        # Calculate pairwise distances
        n = embeddings.size(0)
        dist_matrix = torch.zeros((n, n))
        for i in range(n):
            dist_matrix[i] = torch.norm(embeddings - embeddings[i].unsqueeze(0), p=2, dim=1)
        
        # Group samples by demographic
        demographics = []
        for label in labels.numpy():
            # Get demographic from map, default to 'unknown'
            demographics.append(self.demographics_map.get(label.item(), 'unknown'))
        
        demographics = np.array(demographics)
        unique_demographics = np.unique(demographics)
        
        # Calculate metrics by demographic
        results = {
            'overall': {},
            'by_demographic': {}
        }
        
        # Overall metrics
        correct = 0
        total = 0
        thresholds = np.arange(0.1, 1.6, 0.1)
        far_rates = []
        frr_rates = []
        
        # Calculate FAR/FRR at different thresholds for overall
        for threshold in thresholds:
            far, frr = self._calculate_far_frr(dist_matrix, labels.numpy(), threshold)
            far_rates.append(far)
            frr_rates.append(frr)
        
        # Find EER (point where FAR = FRR)
        differences = np.abs(np.array(far_rates) - np.array(frr_rates))
        eer_idx = np.argmin(differences)
        eer = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        results['overall'] = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'far_curve': far_rates,
            'frr_curve': frr_rates,
            'thresholds': thresholds.tolist()
        }
        
        # Calculate metrics by demographic
        for demo in unique_demographics:
            demo_indices = np.where(demographics == demo)[0]
            if len(demo_indices) < 2:  # Need at least 2 samples per demographic
                continue
                
            # Calculate metrics for this demographic
            demo_dist_matrix = dist_matrix[np.ix_(demo_indices, demo_indices)]
            demo_labels = labels[demo_indices].numpy()
            
            # Calculate FAR/FRR curves
            far_rates = []
            frr_rates = []
            for threshold in thresholds:
                far, frr = self._calculate_far_frr(demo_dist_matrix, demo_labels, threshold)
                far_rates.append(far)
                frr_rates.append(frr)
            
            # Find EER for this demographic
            differences = np.abs(np.array(far_rates) - np.array(frr_rates))
            eer_idx = np.argmin(differences)
            eer = (far_rates[eer_idx] + frr_rates[eer_idx]) / 2
            eer_threshold = thresholds[eer_idx]
            
            # Store results
            results['by_demographic'][demo] = {
                'eer': eer,
                'eer_threshold': eer_threshold,
                'far_curve': far_rates,
                'frr_curve': frr_rates,
                'sample_count': len(demo_indices)
            }
        
        # Visualize and save results
        self._visualize_demographic_results(results, epoch)
        
        # Save results to file
        results_file = os.path.join(self.output_dir, 'demographic_eval', f'results_epoch_{epoch}.json')
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            import json
            serializable_results = self._make_json_serializable(results)
            json.dump(serializable_results, f, indent=4)
        
        return results
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def _calculate_far_frr(self, dist_matrix, labels, threshold):
        """Calculate FAR and FRR for a given threshold"""
        n = len(labels)
        false_accepts = 0
        false_rejects = 0
        total_genuine = 0
        total_impostor = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if labels[i] == labels[j]:  # Same identity (genuine pair)
                    total_genuine += 1
                    if dist_matrix[i, j] > threshold:  # False rejection
                        false_rejects += 1
                else:  # Different identity (impostor pair)
                    total_impostor += 1
                    if dist_matrix[i, j] <= threshold:  # False acceptance
                        false_accepts += 1
        
        far = false_accepts / max(total_impostor, 1)
        frr = false_rejects / max(total_genuine, 1)
        
        return far, frr
    
    def _visualize_demographic_results(self, results, epoch):
        """Visualize and save demographic evaluation results"""
        # Plot 1: EER by demographic group
        plt.figure(figsize=(12, 6))
        
        # Extract demographic groups and their EERs
        demos = list(results['by_demographic'].keys())
        eers = [results['by_demographic'][d]['eer'] for d in demos]
        samples = [results['by_demographic'][d]['sample_count'] for d in demos]
        
        # Sort by EER for better visualization
        sorted_indices = np.argsort(eers)
        demos = [demos[i] for i in sorted_indices]
        eers = [eers[i] for i in sorted_indices]
        samples = [samples[i] for i in sorted_indices]
        
        # Plot EER bars
        bars = plt.bar(demos, eers, color='skyblue')
        
        # Add sample count as text on bars
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'n={samples[i]}', ha='center', va='bottom', fontsize=9)
        
        # Add overall EER as horizontal line
        plt.axhline(y=results['overall']['eer'], color='r', linestyle='-', 
                   label=f'Overall EER: {results["overall"]["eer"]:.3f}')
        
        plt.xlabel('Demographic Group')
        plt.ylabel('Equal Error Rate (EER)')
        plt.title(f'EER by Demographic Group - Epoch {epoch}')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, min(max(eers) * 1.2, 1.0))
        plt.legend()
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'demographic_eval', 
                               f'eer_by_demographic_epoch_{epoch}.png'))
        plt.close()
        
        # Plot 2: DET curves for overall and each demographic
        plt.figure(figsize=(10, 8))
        
        # Plot overall DET curve
        plt.plot(results['overall']['far_curve'], results['overall']['frr_curve'], 
                'k-', linewidth=2, label='Overall')
        
        # Plot curves for each demographic
        for demo in results['by_demographic']:
            plt.plot(results['by_demographic'][demo]['far_curve'],
                    results['by_demographic'][demo]['frr_curve'],
                    '--', linewidth=1, label=f'{demo} (n={results["by_demographic"][demo]["sample_count"]})')
        
        # Add EER line
        plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='EER Line')
        
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.01, 1)
        plt.ylim(0.01, 1)
        plt.xlabel('False Accept Rate (FAR)')
        plt.ylabel('False Reject Rate (FRR)')
        plt.title(f'DET Curves by Demographic - Epoch {epoch}')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(self.output_dir, 'demographic_eval', 
                               f'det_curves_epoch_{epoch}.png'))
        plt.close()