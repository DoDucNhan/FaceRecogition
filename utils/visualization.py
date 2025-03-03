import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import os

class FeatureVisualizer:
    """Visualize feature activations in the model during training"""
    def __init__(self, model, dataloader, device, output_dir):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.output_dir = output_dir
        os.makedirs(os.path.join(output_dir, 'feature_maps'), exist_ok=True)
        
        # Hook dictionaries
        self.hooks = {}
        self.activations = {}
        
    def _get_activation(self, name):
        """Hook function to register for capturing activations"""
        def hook(model, input, output):
            self.activations[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self):
        """Register hooks on specific layers of interest"""
        # Clear previous hooks
        self.remove_hooks()
        
        # Register new hooks - focus on key blocks
        for name, module in self.model.named_modules():
            # Only hook specific layers of interest for face recognition
            if any(layer_type in name for layer_type in 
                  ['conv2d_1a', 'block35.0', 'block17.0', 'block8.0']):
                self.hooks[name] = module.register_forward_hook(self._get_activation(name))
        
        print(f"Registered {len(self.hooks)} hooks for feature visualization")
                
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks = {}
        
    def visualize_features(self, epoch):
        """Visualize features for a batch of images"""
        # Make sure hooks are registered
        if not self.hooks:
            self.register_hooks()
            
        # Set model to eval mode for visualization
        self.model.eval()
        
        # Get a batch of data
        images, labels = next(iter(self.dataloader))
        images = images.to(self.device)
        
        # Forward pass to capture activations
        with torch.no_grad():
            _ = self.model(images)
        
        # Create the visualization grid
        fig, axes = plt.subplots(len(self.activations), 8, figsize=(20, 4*len(self.activations)))
        if len(self.activations) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        for i, (name, activation) in enumerate(self.activations.items()):
            # Get the first 8 channels from the first image in batch
            act = activation[0].numpy()
            
            # Normalize each feature map individually for better visualization
            for j in range(min(8, act.shape[0])):
                feature_map = act[j]
                if feature_map.shape[0] > 0:  # Ensure the feature map is not empty
                    # Normalize to [0, 1]
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
                    axes[i, j].imshow(feature_map, cmap='viridis')
                    axes[i, j].axis('off')
            
            axes[i, 0].set_ylabel(name.split('.')[-2], rotation=90, size='large')
        
        plt.tight_layout()
        plt.suptitle(f'Feature Maps at Epoch {epoch}', fontsize=16)
        plt.subplots_adjust(top=0.92)
        save_path = os.path.join(self.output_dir, 'feature_maps', f'features_epoch_{epoch}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        # Set model back to train mode
        self.model.train()
        
        print(f"Feature visualization saved to {save_path}")
        return save_path