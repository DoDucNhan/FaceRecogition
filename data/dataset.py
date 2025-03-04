import os
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class FaceDataset(Dataset):
    """Dataset class for facial recognition dataset"""
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
            
        # Apply transformation
        if self.transform:
            img = self.transform(img)
                
        return img, label


def is_valid_image(img_path):
    """
    Validate if an image can be properly loaded and is of usable quality
    
    Args:
        img_path: Path to the image file
        
    Returns:
        bool: True if image is valid, False otherwise
    """
    try:
        with Image.open(img_path) as img:
            # Check if image is valid
            img.verify()
            
            # Open image again to check if it can be loaded properly
            img = Image.open(img_path)
            
            # Convert to RGB (some images might be grayscale or RGBA)
            img = img.convert('RGB')
            
            # Check minimum resolution (80x80)
            width, height = img.size
            if width < 80 or height < 80:
                return False
            
            return True
    except Exception:
        return False


def load_dataset(data_dir, min_images_per_person=2, validate_images=True):
    """
    Load the facial dataset from directory structure with image validation
    
    Args:
        data_dir: Path to dataset directory
        min_images_per_person: Minimum number of images required per person
        validate_images: Whether to validate images before adding them
        
    Returns:
        image_paths: List of paths to valid images
        labels: List of numeric labels corresponding to each image
        label_to_idx: Dictionary mapping person IDs to numeric labels
    """
    image_paths = []
    labels = []
    label_to_idx = {}
    
    # List all person directories
    person_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Statistics tracking
    total_images = 0
    valid_images = 0
    skipped_persons = 0
    
    for person_id in tqdm(person_dirs, desc="Loading dataset"):
        person_dir = os.path.join(data_dir, person_id)
        
        # Get all potential image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        potential_images = [f for f in os.listdir(person_dir) 
                          if os.path.splitext(f.lower())[1] in valid_extensions]
        
        total_images += len(potential_images)
        
        # Filter valid images if validation is enabled
        if validate_images:
            valid_image_files = []
            for img_name in potential_images:
                img_path = os.path.join(person_dir, img_name)
                if is_valid_image(img_path):
                    valid_image_files.append(img_name)
                    valid_images += 1
        else:
            valid_image_files = potential_images
            valid_images += len(potential_images)
        
        # Skip people with too few valid images
        if len(valid_image_files) < min_images_per_person:
            skipped_persons += 1
            continue
        
        # Assign a numeric label to this person
        if person_id not in label_to_idx:
            idx = len(label_to_idx)
            label_to_idx[person_id] = idx
        
        # Add all valid images for this person
        for img_name in valid_image_files:
            img_path = os.path.join(person_dir, img_name)
            image_paths.append(img_path)
            labels.append(label_to_idx[person_id])
    
    print(f"Loaded {len(image_paths)} valid images from {len(label_to_idx)} people")
    print(f"Filtered out {total_images - valid_images} invalid images")
    print(f"Skipped {skipped_persons} people with insufficient valid images")
    
    return image_paths, labels, label_to_idx