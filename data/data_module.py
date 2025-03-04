from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import os
import json

from .dataset import FaceDataset, load_dataset


class FaceDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for face datasets"""
    
    def __init__(self, 
                 data_dir, 
                 batch_size=32, 
                 num_workers=2,
                 min_images_per_person=2,
                 validate_images=True,
                 val_split=0.2,
                 seed=42,
                 output_dir=None):
        """
        Initialize face data module
        
        Args:
            data_dir: Path to dataset directory
            batch_size: Batch size for training and validation
            num_workers: Number of workers for data loading
            min_images_per_person: Minimum number of images required per person
            validate_images: Whether to validate images before adding them
            val_split: Validation split ratio
            seed: Random seed for reproducibility
            output_dir: Directory to save label mappings
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.min_images_per_person = min_images_per_person
        self.validate_images = validate_images
        self.val_split = val_split
        self.seed = seed
        self.output_dir = output_dir
        
        self.train_dataset = None
        self.val_dataset = None
        self.label_to_idx = None
        
        # Data transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


    def prepare_data(self):
        """
        Data preparation (called only on one GPU)
        Use this to download or process data that should be done once
        """
        # Nothing to do here since we're using a local dataset
        pass
    

    def setup(self, stage=None):
        """
        Data setup (called on every GPU)
        Use this to split data, define datasets, etc.
        """
        # Load the full dataset
        image_paths, labels, label_to_idx = load_dataset(
            self.data_dir, 
            min_images_per_person=self.min_images_per_person,
            validate_images=self.validate_images
        )
        
        # Save label mappings if output_dir is provided
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, 'label_to_idx.json'), 'w') as f:
                json.dump(label_to_idx, f)
        
        # Split into train and validation sets
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, test_size=self.val_split, 
            random_state=self.seed, stratify=labels
        )
        
        # Create datasets
        self.train_dataset = FaceDataset(train_paths, train_labels, transform=self.train_transform)
        self.val_dataset = FaceDataset(val_paths, val_labels, transform=self.val_transform)
        self.label_to_idx = label_to_idx
        
        print(f"Train set: {len(train_paths)} images")
        print(f"Validation set: {len(val_paths)} images")
    

    def train_dataloader(self):
        """Return training dataloader"""
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    

    def val_dataloader(self):
        """Return validation dataloader"""
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    
    def get_num_classes(self):
        """Return number of classes"""
        return len(self.label_to_idx) if self.label_to_idx else 0