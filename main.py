import os
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Sampler
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# Import from our modules
from utils.visualization import FeatureVisualizer
from utils.demographic_eval import DemographicEvaluator
from data.dataset import load_afd_dataset, AFDDataset
from model.facenet import get_model, freeze_layers, get_layer_groups, print_trainable_parameters
from training.train import train_epoch, evaluate, train_epoch_arcface, evaluate_arcface
from model.loss import TripletLoss, ArcFaceLoss
from training.checkpoint import save_checkpoint, resume_from_checkpoint, load_metrics_for_logger
from utils.helpers import set_seed, ensure_dir
from utils.logger import TrainingLogger

# torch.set_default_dtype(torch.float16)

class IdentityBatchSampler(Sampler):
    def __init__(self, labels, batch_size, faces_per_identity):
        self.labels = labels
        self.batch_size = batch_size
        self.faces_per_identity = faces_per_identity
        self.label_to_indices = {}
        
        # Group indices by label
        for idx, label in enumerate(labels):
            if label not in self.label_to_indices:
                self.label_to_indices[label] = []
            self.label_to_indices[label].append(idx)
            
        # Calculate number of batches per epoch
        self.num_identities_per_batch = self.batch_size // self.faces_per_identity
        self.num_batches = len(self.label_to_indices) // self.num_identities_per_batch
    
    def __iter__(self):
        # Create batches for one epoch
        for _ in range(self.num_batches):
            batch = []
            # Sample identities
            identities = np.random.choice(
                list(self.label_to_indices.keys()), 
                self.num_identities_per_batch, 
                replace=False
            )
            
            # Sample faces for each identity
            for identity in identities:
                indices = self.label_to_indices[identity]
                if len(indices) >= self.faces_per_identity:
                    batch.extend(np.random.choice(indices, 
                               self.faces_per_identity, replace=False))
                else:
                    batch.extend(np.random.choice(indices, 
                               self.faces_per_identity, replace=True))
            
            yield batch
    
    def __len__(self):
        return self.num_batches



def main():
    parser = argparse.ArgumentParser(description="FaceNet Finetuning")
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--min_images', type=int, default=2, help='Minimum images per person')
    parser.add_argument('--validate_images', action='store_true', help='Validate images before training')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, default=10, 
                        help='Patience for early stopping (epochs with no improvement before stopping, 0 to disable)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.2, help='Margin for triplet loss')
    parser.add_argument('--mining', choices=['batch_all', 'batch_hard'], default='batch_hard', 
                        help='Triplet mining strategy')
    parser.add_argument('--faces_per_identity', type=int, default=40, 
                        help='Number of faces per identity in each batch')
    # Arguments for ArcFace+UNPG
    parser.add_argument('--use_arcface', action='store_true', 
                        help='Use ArcFace loss instead of triplet loss')
    parser.add_argument('--use_unpg', action='store_true', 
                        help='Use Unified Negative Pair Generation with ArcFace')
    parser.add_argument('--arcface_scale', type=float, default=64.0, 
                        help='Scale factor for ArcFace loss')
    parser.add_argument('--arcface_margin', type=float, default=0.5, 
                        help='Angular margin for ArcFace loss')
    parser.add_argument('--wisker_size', type=float, default=1.5, 
                        help='Wisker size for UNPG filtering')
    parser.add_argument('--grad_accumulation', type=int, default=2, 
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training to save memory')
    
    # Visualization parameters
    parser.add_argument('--visualize_features', action='store_true', 
                        help='Whether to visualize feature maps during training')
    parser.add_argument('--demographic_eval', action='store_true',
                        help='Whether to perform demographic evaluation')
    parser.add_argument('--demographics_file', type=str, default=None,
                        help='Path to file mapping label_ids to demographic groups')
    parser.add_argument('--vis_frequency', type=int, default=5,
                        help='Frequency (in epochs) for feature visualization')
    
    #Logging parameters
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str, default="facenet-tuning", help='Name for this experiment')
    
    # Freezing parameters
    parser.add_argument('--freeze_groups', type=int, default=5, help='Initial number of groups to freeze')
    parser.add_argument('--unfreeze_epoch', type=int, default=5, help='Unfreeze a group every N epochs')
    
    # Learning rate parameters
    parser.add_argument('--lr_scheduler', choices=['cosine', 'plateau', 'sgdr'], default='cosine',
                        help='Learning rate scheduler')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate')
    
    # Checkpoint parameters
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    parser.add_argument('--save_freq', type=int, default=1, help='Save checkpoint frequency (epochs)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of worker threads for dataloader')
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_dir(args.output_dir)
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Logger
    logger = TrainingLogger(args.log_dir, experiment_name=args.experiment_name)
    
    # Data transformations
    train_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the full dataset
    print(f"Loading dataset from {args.data_dir}...")
    image_paths, labels, label_to_idx = load_afd_dataset(
        args.data_dir, 
        min_images_per_person=args.min_images,
        validate_images=args.validate_images
    )
    
    # Get model
    if args.use_arcface:
        num_classes = len(label_to_idx)
        model = get_model(pretrained=args.pretrained, device=device, 
                        use_arcface=True, num_classes=num_classes)
        
        # ArcFace loss with optional UNPG
        criterion = ArcFaceLoss(
            embedding_size=512,
            num_classes=num_classes,
            s=args.arcface_scale,
            m=args.arcface_margin,
            use_unpg=args.use_unpg,
            wisker_size=args.wisker_size
        ).to(device)
    else:
        # Original triplet loss setup
        model = get_model(pretrained=args.pretrained, device=device)
        criterion = TripletLoss(margin=args.margin)
    
    # Variables to track training
    start_epoch = 1
    best_accuracy = 0.0
    prev_accuracy = 0.0
    current_frozen_groups = args.freeze_groups
    train_paths = None
    train_labels = None
    val_paths = None
    val_labels = None
    
    # Initialize optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
    
    # Resume from checkpoint if specified
    if args.resume and os.path.isfile(args.resume):
        load_metrics_for_logger(logger, args.resume)
        # Initialize criterion before resuming if using ArcFace
        if args.use_arcface:
            num_classes = len(label_to_idx) if label_to_idx else 0  # Will be updated by resume
            criterion = ArcFaceLoss(
                embedding_size=512,
                num_classes=num_classes,
                s=args.arcface_scale,
                m=args.arcface_margin,
                use_unpg=args.use_unpg,
                wisker_size=args.wisker_size
            ).to(device)
        else:
            criterion = TripletLoss(margin=args.margin)

        start_epoch, best_accuracy, loaded_label_to_idx, current_frozen_groups = resume_from_checkpoint(
            args.resume, model, optimizer, 
            criterion=criterion if args.use_arcface else None
        )
        start_epoch += 1  # Start from next epoch
        prev_accuracy = best_accuracy
        print(f"Resumed from epoch {start_epoch-1} with accuracy: {best_accuracy:.4f}")
        # Update label_to_idx from checkpoint
        label_to_idx = loaded_label_to_idx
        
        # If using ArcFace, update the criterion's num_classes in case it changed
        if args.use_arcface and len(label_to_idx) != criterion.num_classes:
            print(f"Updating ArcFace criterion with {len(label_to_idx)} classes")
            criterion = ArcFaceLoss(
                embedding_size=512,
                num_classes=len(label_to_idx),
                s=args.arcface_scale,
                m=args.arcface_margin,
                use_unpg=args.use_unpg,
                wisker_size=args.wisker_size
            ).to(device)
            
            # If the checkpoint has ArcFace state, load it
            checkpoint = torch.load(args.resume)
            if 'arcface_state' in checkpoint and checkpoint['arcface_state'] is not None:
                criterion.load_state_dict(checkpoint['arcface_state'])
        
        
        # # We'll use the label_to_idx from checkpoint, but verify it's compatible with the dataset
        # if set(label_to_idx.keys()) != set(label_to_idx.keys()):
        #     print("Warning: The dataset has changed since the checkpoint was created.")
        #     print(f"Checkpoint has {len(label_to_idx)} identities, current dataset has {len(label_to_idx)} identities.")
        
        # Recompute labels based on the checkpoint's label_to_idx
        labels = [label_to_idx.get(person_id, -1) for person_id in [list(label_to_idx.keys())[l] for l in labels]]
        
        # Filter out any data points with invalid labels (person not in checkpoint's label_to_idx)
        valid_indices = [i for i, l in enumerate(labels) if l != -1]
        image_paths = [image_paths[i] for i in valid_indices]
        labels = [labels[i] for i in valid_indices]
        
        # print(f"After filtering to match checkpoint identities: {len(image_paths)} images")
        
        # Split into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=args.seed, stratify=labels
        )
        
        print(f"Resumed from epoch {start_epoch-1} with accuracy: {best_accuracy:.4f}")
        print(f"Train set: {len(train_paths)} images")
        print(f"Validation set: {len(val_paths)} images")
    else:
        # Save label mappings
        with open(os.path.join(args.output_dir, 'label_to_idx.json'), 'w') as f:
            json.dump(label_to_idx, f)
        
        # Split into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=0.2, random_state=args.seed, stratify=labels
        )
        
        print(f"Train set: {len(train_paths)} images")
        print(f"Validation set: {len(val_paths)} images")
        
        # Apply initial layer freezing
        current_frozen_groups = freeze_layers(model, current_frozen_groups)
    
    # Create datasets and dataloaders
    train_dataset = AFDDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = AFDDataset(val_paths, val_labels, transform=val_transform)
    
    # For ArcFace+UNPG, we need standard batching
    print(f"Using ArcFace: {args.use_arcface}, Using UNPG: {args.use_unpg}")
    if args.use_arcface:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(args.num_workers, 2),  # Reduce workers to save memory
            pin_memory=True
        )
    else:
        # Original triplet mining approach
        train_loader = DataLoader(
            train_dataset, 
            batch_sampler=IdentityBatchSampler(
                train_labels, 
                args.batch_size, 
                args.faces_per_identity
            ),
            num_workers=min(args.num_workers, 2),
            pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=min(args.num_workers, 2),
        pin_memory=True
    )

    # Initial freezing - freeze all except last 2 groups
    if not args.resume:
        # For new training, freeze all except the last groups
        # initial_unfrozen_groups = 1
        # current_frozen_groups = total_groups - initial_unfrozen_groups
        print(f"Initially freezing {current_frozen_groups} groups")
        current_frozen_groups = freeze_layers(model, current_frozen_groups)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5)
        print_trainable_parameters(model)
    
    # Initialize scheduler
    if args.lr_scheduler == 'sgdr':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=args.lr_min)
    elif args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    else:  # plateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3, 
            min_lr=args.lr_min, verbose=True)

    # Initialize tracking variables for adaptive unfreezing
    adaptive_patience_counter = 0
    prev_accuracy = 0

    # Get initial layer groups
    layer_groups = get_layer_groups(model)
    total_groups = len(layer_groups)

    # Early stopping tracker
    patience_counter = 0
    # Initialize mixed precision scaler if enabled
    scaler = torch.amp.GradScaler('cuda') if args.mixed_precision else None

    #------------------------------------------------------------------------
    # Initialize feature visualizer if enabled
    feature_visualizer = None
    if args.visualize_features:
        feature_visualizer = FeatureVisualizer(model, val_loader, device, args.output_dir)

    # Initialize demographic evaluator if enabled
    demographic_evaluator = None
    if args.demographic_eval and args.demographics_file:
        # Load demographics map from file
        demographics_map = {}
        try:
            with open(args.demographics_file, 'r') as f:
                demographics_map = json.load(f)
            print(f"Loaded demographics data for {len(demographics_map)} identities")
            demographic_evaluator = DemographicEvaluator(
                model, val_loader, demographics_map, device, args.output_dir)
        except Exception as e:
            print(f"Error loading demographics file: {str(e)}")
            print("Demographic evaluation will be disabled")
    #------------------------------------------------------------------------

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Visualize features before training if enabled
        if feature_visualizer and epoch % args.vis_frequency == 0:
            feature_visualizer.visualize_features(epoch)
        
        # Gradual unfreezing
        if ((args.unfreeze_epoch > 0 and epoch % args.unfreeze_epoch == 0 ) or adaptive_patience_counter >= 20) and current_frozen_groups > 0:
            previous_frozen_groups = current_frozen_groups
            current_frozen_groups -= 1
            print(f"\n{'='*50}")
            print(f"Unfreezing next layer group, {current_frozen_groups} groups remain frozen")
            current_frozen_groups = freeze_layers(model, current_frozen_groups)
            print_trainable_parameters(model)
                
            layer_groups = get_layer_groups(model)
            # Identify newly unfrozen groups
            newly_unfrozen_groups = []
            for i in range(current_frozen_groups, previous_frozen_groups):
                if i < len(layer_groups):
                    newly_unfrozen_groups.extend(layer_groups[i])

            # Collect newly unfrozen parameters
            newly_unfrozen_params = []
            for name, param in model.named_parameters():
                if any(layer_name in name for layer_name in newly_unfrozen_groups) and param.requires_grad:
                    newly_unfrozen_params.append(param)

            optimizer.add_param_group({
                'params': newly_unfrozen_params,
                'weight_decay': optimizer.param_groups[0]['weight_decay']
            })
            # Reset counter
            adaptive_patience_counter = 0
            
            # Implement warm restart for the optimizer
            if args.lr_scheduler == 'sgdr':
                # SGDR will handle the restart automatically on next call to step()
                pass
            else:
                # Manual restart - reduce LR then reschedule
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * 0.5  # Restart with lower LR
                
                # Recreate scheduler with remaining epochs
                remaining_epochs = args.epochs - epoch
                if args.lr_scheduler == 'cosine':
                    scheduler = CosineAnnealingLR(
                        optimizer, T_max=remaining_epochs, eta_min=args.lr_min)
                else:  # plateau
                    scheduler = ReduceLROnPlateau(
                        optimizer, mode='max', factor=0.5, patience=3, 
                        min_lr=args.lr_min, verbose=True)
                    
            print(f"Learning rate after restart: {optimizer.param_groups[0]['lr']:.8f}")
            print(f"{'='*50}\n")


        # Train for one epoch
        if args.use_arcface:
            train_loss = train_epoch_arcface(
                model, train_loader, criterion, optimizer, device, epoch,
                grad_accumulation_steps=args.grad_accumulation,
                scaler=scaler
            )
        else:
            train_loss = train_epoch(
                model, train_loader, optimizer, device, epoch, 
                margin=args.margin, mining_method=args.mining
            )
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        # Clear memory before evaluation
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        
        # Evaluate on validation set
        if args.use_arcface:
            accuracy, far = evaluate_arcface(model, val_loader, device)
        else:
            accuracy, far = evaluate(model, val_loader, device)
        print(f"Validation Accuracy: {accuracy:.4f}")
        print(f"Validation FAR: {far:.4f}")

        # Perform demographic evaluation if enabled
        if demographic_evaluator and epoch % args.vis_frequency == 0:
            demo_results = demographic_evaluator.evaluate(epoch)
            print("Demographic evaluation completed")
        
        # Update scheduler
        if args.lr_scheduler == 'cosine' or args.lr_scheduler == 'sgdr':
            scheduler.step()
        else:  # plateau
            scheduler.step(accuracy)
        
        # Save model if it's the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            # Clear memory before saving
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': accuracy,
                'label_to_idx': label_to_idx
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with accuracy: {best_accuracy:.4f}")
        
        # Check if accuracy has improved
        if accuracy > prev_accuracy + 0.001:  # 0.1% improvement threshold
            layer_patience_counter = 0
            patience_counter = 0
        else:
            layer_patience_counter += 1
            # Increment patience counter if no improvement
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
            
            # Check if we should stop training
            if args.early_stopping > 0 and patience_counter >= args.early_stopping:
                print(f"Early stopping after {patience_counter} epochs without improvement")
                break


        accuracy_delta = accuracy - prev_accuracy
        if accuracy_delta < 0.001:  # If improvement is minimal
            adaptive_patience_counter += 1
            print(f"Limited improvement: {accuracy_delta:.4f}, patience counter: {adaptive_patience_counter}")
        else:
            # Good improvement, reset patience counter
            adaptive_patience_counter = 0
        
        
        prev_accuracy = accuracy
        
        # Save checkpoint for resumption
        if epoch % args.save_freq == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, prev_accuracy,
                label_to_idx, current_frozen_groups, args,
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
        
        # Always save latest checkpoint for unexpected interruptions
        save_checkpoint(
            model, optimizer, scheduler, epoch, prev_accuracy,
            label_to_idx, current_frozen_groups, args,
            os.path.join(args.output_dir, 'latest_checkpoint.pth')
        )
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.8f}")

        # Log metrics
        logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_accuracy=accuracy,
            val_far=far,
            learning_rate=current_lr,
            extra_metrics={
                'unfrozen_groups': total_groups - current_frozen_groups
            }
        )
    
    print(f"Training completed! Best accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()