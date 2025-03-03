import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1

def get_model(pretrained=False, device=None, use_arcface=False, num_classes=None):
    """
    Initialize FaceNet model with optional ArcFace integration
    
    Args:
        pretrained: Whether to use pretrained weights
        device: Device to put model on
        use_arcface: Whether to use ArcFace model structure
        num_classes: Number of identity classes (needed for ArcFace)
        
    Returns:
        model: Initialized FaceNet model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_arcface and num_classes is None:
        raise ValueError("Number of classes must be provided when using ArcFace")
    
    # Get base FaceNet model
    if pretrained:
        print("Loading pretrained FaceNet model...")
        base_model = InceptionResnetV1(pretrained='casia-webface').to(device)
    else:
        print("Initializing FaceNet model from scratch...")
        base_model = InceptionResnetV1(classify=False).to(device)
        
    # Wrap with ArcFace if needed
    if use_arcface:
        print(f"Setting up model for ArcFace with {num_classes} identity classes")
        model = ArcFaceWrapper(base_model, num_classes).to(device)
    else:
        model = base_model
    
    return model


class ArcFaceWrapper(nn.Module):
    """Wrapper for FaceNet model to use with ArcFace loss"""
    def __init__(self, base_model, num_classes):
        super(ArcFaceWrapper, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
    
    def forward(self, x):
        """Extract feature embeddings"""
        embeddings = self.base_model(x)
        return embeddings


# Model 128
def get_model_128(pretrained=False, device=None, use_arcface=False, num_classes=None):
    """
    Initialize FaceNet model with optional ArcFace integration
    
    Args:
        pretrained: Whether to use pretrained weights
        device: Device to put model on
        use_arcface: Whether to use ArcFace model structure
        num_classes: Number of identity classes (needed for ArcFace)
        
    Returns:
        model: Initialized FaceNet model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if use_arcface and num_classes is None:
        raise ValueError("Number of classes must be provided when using ArcFace")
    
    # Get base FaceNet model
    if pretrained:
        print("Loading pretrained FaceNet model...")
        base_model = torch.load("facenet-128.pt").to(device)
    else:
        raise ValueError("Must use pretrained model...")
        
    # Wrap with ArcFace if needed
    if use_arcface:
        print(f"Setting up model for ArcFace with {num_classes} identity classes")
        model = ArcFaceWrapper(base_model, num_classes).to(device)
    else:
        model = base_model
    
    return model


def get_layer_groups(model):
    """
    Divide model into layer groups for gradual unfreezing
    
    Args:
        model: FaceNet model
        
    Returns:
        groups: List of lists of layer names
    """
    # For InceptionResnetV1, divide the model into specific functional blocks
    layer_names = [name for name, _ in model.named_parameters()]
    
    # Group by block with more fine-grained control
    blocks = {
        'logits': [],      # Final classification layer
        'last_conv': [],   # Last convolutional layers before global pooling
        'block8': [],      # Inception Block 8 layers
        'reduction_b': [], # Reduction layers between Block17 and Block8
        'block17': [],     # Inception Block 17 layers
        'reduction_a': [], # Reduction layers between Block35 and Block17
        'block35': [],     # Inception Block 35 layers
        'initial': [],     # Initial layers
    }
    
    for name in layer_names:
        if 'logits' in name or 'classifier' in name:
            blocks['logits'].append(name)
        elif 'last_conv' in name or 'last_bn' in name or 'conv2d_7b' in name:
            blocks['last_conv'].append(name)
        elif 'block8' in name:
            blocks['block8'].append(name)
        elif 'reduction_b' in name:
            blocks['reduction_b'].append(name)
        elif 'block17' in name:
            blocks['block17'].append(name)
        elif 'reduction_a' in name:
            blocks['reduction_a'].append(name)
        elif 'block35' in name:
            blocks['block35'].append(name)
        elif any(x in name for x in ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'stem']):
            blocks['initial'].append(name)
        else:
            # Add any uncategorized layers to the last_conv group as a fallback
            blocks['last_conv'].append(name)
    
    # Create ordered groups from bottom (logits) to top (initial)
    ordered_groups = [
        blocks['logits'],
        blocks['last_conv'],
        blocks['block8'],
        blocks['reduction_b'],
        blocks['block17'],
        blocks['reduction_a'],
        blocks['block35'],
        blocks['initial']
    ]
    
    # Filter out empty groups
    groups = [group for group in ordered_groups if group]
    
    return groups


def freeze_layers(model, num_groups_to_freeze, verbose=True):
    """
    Freeze specified number of layer groups from the earliest layers
    
    Args:
        model: FaceNet model
        num_groups_to_freeze: Number of layer groups to freeze (counting from earliest layers)
        verbose: Whether to print verbose information about freezing
        
    Returns:
        num_groups_frozen: Actual number of groups frozen
    """
    # Get layer groups
    layer_groups = get_layer_groups(model)
    total_groups = len(layer_groups)
    
    if verbose:
        print(f"Model has {total_groups} layer groups")
    
    # Make sure num_groups_to_freeze is valid
    num_groups_to_freeze = min(num_groups_to_freeze, total_groups-1)
    
    # First unfreeze all layers
    for param in model.parameters():
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
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layers_to_freeze):
            param.requires_grad = False
            frozen_count += 1
    
    if verbose:
        # Print frozen layers info
        total_params = sum(1 for _ in model.parameters())
        print(f"Froze {frozen_count}/{total_params} parameters ({frozen_count/total_params:.2%})")
        print(f"Froze {num_groups_to_freeze}/{total_groups} layer groups")
    
    return num_groups_to_freeze


def get_trainable_parameters(model):
    """
    Count and summarize trainable parameters by layer group
    
    Args:
        model: FaceNet model
        
    Returns:
        summary: Dictionary with summary of trainable parameters by group
    """
    layer_groups = get_layer_groups(model)
    
    # Initialize summary
    summary = {
        'total_params': 0,
        'trainable_params': 0,
        'frozen_params': 0,
        'groups': {}
    }
    
    # Count total parameters
    summary['total_params'] = sum(p.numel() for p in model.parameters())
    
    # Count trainable parameters
    summary['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary['frozen_params'] = summary['total_params'] - summary['trainable_params']
    
    # Count parameters by group
    for i, group in enumerate(layer_groups):
        group_name = f"Group {i}" if i < len(layer_groups) else "Other"
        
        trainable_params = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in group):
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
        
        summary['groups'][group_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'frozen_params': total_params - trainable_params,
            'trainable_percentage': trainable_params / total_params if total_params > 0 else 0
        }
    
    return summary


def print_trainable_parameters(model):
    """
    Print a summary of trainable parameters by layer group
    
    Args:
        model: FaceNet model
    """
    summary = get_trainable_parameters(model)
    
    print("=" * 50)
    print("Trainable Parameters Summary")
    print("=" * 50)
    print(f"Total parameters: {summary['total_params']:,}")
    print(f"Trainable parameters: {summary['trainable_params']:,} ({summary['trainable_params']/summary['total_params']:.2%})")
    print(f"Frozen parameters: {summary['frozen_params']:,} ({summary['frozen_params']/summary['total_params']:.2%})")
    print("-" * 50)
    print("Parameters by layer group:")
    
    for group_name, group_stats in summary['groups'].items():
        print(f"{group_name}:")
        print(f"  Total: {group_stats['total_params']:,}")
        print(f"  Trainable: {group_stats['trainable_params']:,} ({group_stats['trainable_percentage']:.2%})")
        print(f"  Frozen: {group_stats['frozen_params']:,}")
    
    print("=" * 50)