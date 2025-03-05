import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import pytorch_lightning as pl
from facenet_pytorch import InceptionResnetV1
from .loss import ArcFaceLoss


class FaceNetEmbedding(nn.Module):
    def __init__(self, original_model):
        super(FaceNetEmbedding, self).__init__()
        # Copy all layers except the final logits layer
        self.conv2d_1a = original_model.conv2d_1a
        self.conv2d_2a = original_model.conv2d_2a
        self.conv2d_2b = original_model.conv2d_2b
        self.maxpool_3a = original_model.maxpool_3a
        self.conv2d_3b = original_model.conv2d_3b
        self.conv2d_4a = original_model.conv2d_4a
        self.conv2d_4b = original_model.conv2d_4b
        self.repeat_1 = original_model.repeat_1
        self.mixed_6a = original_model.mixed_6a
        self.repeat_2 = original_model.repeat_2
        self.mixed_7a = original_model.mixed_7a
        self.repeat_3 = original_model.repeat_3
        self.block8 = original_model.block8
        self.avgpool_1a = original_model.avgpool_1a
        self.dropout = original_model.dropout
        self.last_linear = original_model.last_linear
        self.last_bn = original_model.last_bn
        
    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        x = self.last_bn(x)
        return x


class FaceNetLightning(pl.LightningModule):
    """PyTorch Lightning module for FaceNet with gradual unfreezing"""
    
    def __init__(self, 
                 num_classes, 
                 pretrained=False, 
                 learning_rate=0.005,
                 lr_scheduler='cosine',
                 lr_min=1e-6,
                 use_unpg=False,
                 arcface_scale=64.0,
                 arcface_margin=0.5,
                 wisker_size=1.5,
                 freeze_initial_layers=5,
                 unfreeze_epoch_freq=5,
                 weight_decay=1e-5,
                 mixed_precision=False,
                 resume_from_checkpoint=None):
        """
        Initialize FaceNet Lightning Module
        
        Args:
            num_classes: Number of identity classes
            pretrained: Whether to use pretrained model
            learning_rate: Initial learning rate
            lr_scheduler: Type of learning rate scheduler ('cosine', 'plateau', or 'sgdr')
            lr_min: Minimum learning rate
            use_unpg: Whether to use UNPG with ArcFace
            arcface_scale: Scale factor for ArcFace
            arcface_margin: Angular margin for ArcFace
            wisker_size: Wisker size for UNPG filtering
            freeze_initial_layers: Number of initial layer groups to freeze
            unfreeze_epoch_freq: Frequency (in epochs) to unfreeze layers
            weight_decay: Weight decay for optimizer
            mixed_precision: Whether to use mixed precision training
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Model configuration
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_initial_layers = freeze_initial_layers
        self.unfreeze_epoch_freq = unfreeze_epoch_freq
        self.current_frozen_groups = freeze_initial_layers
        self.adaptive_patience_counter = 20
        self.prev_accuracy = 0.0
        
        # Training configuration
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_min = lr_min
        self.weight_decay = weight_decay
        self.mixed_precision = mixed_precision
        self.resume_from_checkpoint = resume_from_checkpoint
        
        # Initialize model
        if self.pretrained:
            pretrained_model = InceptionResnetV1(classify=False, pretrained='casia-webface')
            # Remove the last logits layer
            self.model = FaceNetEmbedding(pretrained_model)
        else:
            self.model = InceptionResnetV1(classify=False, pretrained=None)

        # Initialize ArcFace loss
        self.criterion = ArcFaceLoss(
            embedding_size=512,
            num_classes=num_classes,
            s=arcface_scale,
            m=arcface_margin,
            use_unpg=use_unpg,
            wisker_size=wisker_size
        )
        
        # Apply initial layer freezing if not resuming
        if self.resume_from_checkpoint is None:
            self.current_frozen_groups = self._freeze_layers(self.current_frozen_groups, log=False)
       
        
    def on_load_checkpoint(self, checkpoint):
        """Called when loading a checkpoint - handle resumption appropriately"""
        state_dict = checkpoint['state_dict']

        # Check if we need to update the criterion (number of classes might have changed)
        if hasattr(self.criterion, 'num_classes') and self.criterion.num_classes != self.num_classes:
            self.criterion = ArcFaceLoss(
                embedding_size=512,
                num_classes=self.num_classes,
                s=self.hparams.arcface_scale,
                m=self.hparams.arcface_margin,
                use_unpg=self.hparams.use_unpg,
                wisker_size=self.hparams.wisker_size
            )
    
        # Load current frozen groups state
        if 'current_frozen_groups' in checkpoint:
            self.current_frozen_groups = checkpoint['current_frozen_groups']
            self.current_frozen_groups = self._freeze_layers(self.current_frozen_groups, log=False)
        
        # Load accuracy tracking for adaptive unfreezing
        if 'prev_accuracy' in checkpoint:
            self.prev_accuracy = checkpoint['prev_accuracy']
        
        if 'adaptive_patience_counter' in checkpoint:
            self.adaptive_patience_counter = checkpoint['adaptive_patience_counter']
    

    def on_save_checkpoint(self, checkpoint):
        """Called when saving a checkpoint - add additional state"""
        # Save current frozen groups state
        checkpoint['current_frozen_groups'] = self.current_frozen_groups
        checkpoint['prev_accuracy'] = self.prev_accuracy
        checkpoint['adaptive_patience_counter'] = self.adaptive_patience_counter
    

    def forward(self, x):
        """Forward pass"""
        return self.model(x)
    

    def _get_layer_groups(self):
        """Divide model into layer groups for gradual unfreezing"""
        # Get all parameter names
        layer_names = [name for name, _ in self.model.named_parameters()]
        
        # Group layers according to FaceNet architecture blocks
        groups = [
            # Group 1: Final layers (unfreeze first)
            [name for name in layer_names if any(x in name for x in ['last_linear', 'last_bn'])],
            
            # Group 2: Block8 and final conv layers
            [name for name in layer_names if any(x in name for x in ['block8', 'avgpool_1a', 'dropout'])],
            
            # Group 3: Repeat_3 (Block8 repeat units)
            [name for name in layer_names if 'repeat_3' in name],
            
            # Group 4: Mixed_7a (reduction block)
            [name for name in layer_names if 'mixed_7a' in name],
            
            # Group 5: Repeat_2 (Block17 repeat units)
            [name for name in layer_names if 'repeat_2' in name],
            
            # Group 6: Mixed_6a (reduction block)
            [name for name in layer_names if 'mixed_6a' in name],
            
            # Group 7: Repeat_1 (Block35 repeat units)
            [name for name in layer_names if 'repeat_1' in name],
            
            # Group 8: Initial convolutional layers (freeze first)
            [name for name in layer_names if any(x in name for x in [
                'conv2d_1a', 'conv2d_2a', 'conv2d_2b', 'conv2d_3b', 
                'conv2d_4a', 'conv2d_4b', 'maxpool_3a'
            ])],
        ]
        
        # Filter out empty groups and return
        return [group for group in groups if group]
        

    def _freeze_layers(self, num_groups_to_freeze, log=True):
        """Freeze specified number of layer groups from the earliest layers"""
        # Get layer groups
        layer_groups = self._get_layer_groups()
        total_groups = len(layer_groups)
        
        # Make sure num_groups_to_freeze is valid
        num_groups_to_freeze = min(num_groups_to_freeze, total_groups-1)
        
        # First unfreeze all layers
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Then freeze the specified number of groups starting from the earliest layers
        # Remember our groups are ordered from logits (latest) to initial (earliest)
        groups_to_freeze = layer_groups[-num_groups_to_freeze:] if num_groups_to_freeze > 0 else []
        
        # Flatten the list of layers to freeze
        layers_to_freeze = []
        for group in groups_to_freeze:
            layers_to_freeze.extend(group)
        
        # Freeze layers
        frozen_count = 0
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layers_to_freeze):
                param.requires_grad = False
                frozen_count += 1
        
        # Print frozen layers info
        total_params = sum(1 for _ in self.model.parameters())
        if log and hasattr(self, 'trainer') and self.trainer is not None:
            self.log('frozen_layers_percent', frozen_count/total_params)
            self.log('frozen_groups', num_groups_to_freeze)
            self.log('unfrozen_groups', total_groups - num_groups_to_freeze)
        
        # Print layers frozen for user feedback
        print(f"Froze {frozen_count}/{total_params} parameters ({frozen_count/total_params:.2%})")
        print(f"Froze {num_groups_to_freeze}/{total_groups} layer groups")
        
        return num_groups_to_freeze
    

    def _unfreeze_next_layer_group(self):
        """Unfreeze the next layer group"""
        if self.current_frozen_groups > 0:
            # Store previously frozen parameters before unfreezing
            previously_frozen_params = set()
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    previously_frozen_params.add(param)

            self.current_frozen_groups -= 1
            self.current_frozen_groups = self._freeze_layers(self.current_frozen_groups)

            # Log unfreezing event
            self.log('unfrozen_groups', len(self._get_layer_groups()) - self.current_frozen_groups)
            self.adaptive_patience_counter = 0
            
            print(f"Unfreezing next layer group. {self.current_frozen_groups} groups remain frozen.")
    

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, labels = batch
        
        # Forward pass
        embeddings = self(images)
        loss, logits = self.criterion(embeddings, labels)
        
        # Compute accuracy for logging
        _, predicted = torch.max(logits.data, 1)
        accuracy = (predicted == labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_accuracy', accuracy, on_step=False, on_epoch=True, prog_bar=True)
        
        # Save memory by clearing gradients periodically
        if self.trainer.global_step % 10 == 0 and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
        return loss
    

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, labels = batch
        
        # Get embeddings
        embeddings = self(images)
        
        # Store embeddings and labels for end of epoch
        return {'embeddings': embeddings, 'labels': labels}
    

    # Add this method to collect all validation outputs
    def on_validation_epoch_start(self):
        # Initialize empty lists to store all embeddings and labels
        self.val_embeddings = []
        self.val_labels = []

    # Replace validation_epoch_end with this method
    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        # Collect outputs from each validation batch
        self.val_embeddings.append(outputs['embeddings'])
        self.val_labels.append(outputs['labels'])


    # Add this method to process all validation results
    def on_validation_epoch_end(self):
        """Process validation results at the end of the epoch"""
        # Concatenate all embeddings and labels
        embeddings = torch.cat(self.val_embeddings, dim=0)
        labels = torch.cat(self.val_labels, dim=0)
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute accuracy using nearest neighbor approach
        correct = 0
        total = 0
        
        # For each embedding, find the nearest neighbor
        for i in range(len(embeddings)):
            query_emb = embeddings[i].unsqueeze(0)
            query_label = labels[i].item()
            
            # Compute distances to all other embeddings
            dists = torch.norm(embeddings - query_emb, dim=1, p=2)
            
            # Set distance to self to infinity to exclude it
            dists[i] = float('inf')
            
            # Get nearest neighbor
            _, nn_idx = torch.min(dists, dim=0)
            nn_label = labels[nn_idx].item()
            
            # Check if nearest neighbor has the same label
            if nn_label == query_label:
                correct += 1
            total += 1
        
        accuracy = correct / total
        self.log('val_accuracy', accuracy, prog_bar=True)
        
        # Check for unfreezing layers based on accuracy improvement
        accuracy_delta = accuracy - self.prev_accuracy
        if accuracy_delta < 0.001:  # If improvement is minimal
            self.adaptive_patience_counter += 1
            print(f"\nLimited improvement: {accuracy_delta:.4f}, patience counter: {self.adaptive_patience_counter}")
        else:
            # Good improvement, reset patience counter
            self.adaptive_patience_counter = 0
            print(f"\nGood improvement: {accuracy_delta:.4f}, reset patience counter")
            
        # Unfreeze next layer group if needed
        current_epoch = self.current_epoch + 1  # 0-based to 1-based
        if (self.unfreeze_epoch_freq > 0 and current_epoch % self.unfreeze_epoch_freq == 0) or self.adaptive_patience_counter >= 10:
            self._unfreeze_next_layer_group()
            
        self.prev_accuracy = accuracy
        torch.cuda.empty_cache()
        
        # Clean up to free memory
        del self.val_embeddings
        del self.val_labels
        
        return accuracy
    

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers"""
        # Only include parameters that require gradients
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Configure scheduler
        if self.lr_scheduler == 'sgdr':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer, T_0=20, T_mult=2, eta_min=self.lr_min)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_accuracy'
            }
        elif self.lr_scheduler == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs, eta_min=self.lr_min)
            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                'monitor': 'val_accuracy'
            }
        else:  # plateau
            scheduler = ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, 
                min_lr=self.lr_min, verbose=True)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_accuracy',
                    'interval': 'epoch'
                }
            }