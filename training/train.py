import torch
import torch.nn.functional as F
from tqdm import tqdm
from model.loss import mine_triplets


def compute_far(embeddings, labels, threshold=0.3):
    """Compute False Accept Rate at specific threshold"""
    false_accepts = 0
    total_comparisons = 0

    # Make sure both are on the same device
    labels = labels.to(embeddings.device)
    
    for i in range(len(embeddings)):
        anchor_emb = embeddings[i]
        anchor_label = labels[i]
        
        # Compare with all other embeddings
        distances = torch.norm(embeddings - anchor_emb, p=2, dim=1)
        
        # Count false accepts (different identity but distance below threshold)
        mask = (labels != anchor_label) & (distances < threshold)
        false_accepts += mask.sum().item()
        total_comparisons += (labels != anchor_label).sum().item()
    
    return false_accepts / total_comparisons if total_comparisons > 0 else 1.0


def train_epoch(model, train_loader, optimizer, device, epoch, margin=0.2, mining_method='batch_hard'):
    """
    Train model for one epoch
    
    Args:
        model: FaceNet model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        margin: Margin for triplet loss
        mining_method: Method for mining triplets
        
    Returns:
        avg_loss: Average loss for this epoch
    """
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
    n_valid_batches = 0
    
    progress_bar = tqdm(enumerate(train_loader), total=n_batches, 
                       desc=f"Epoch {epoch}")
    
    for batch_idx, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass to get embeddings
        embeddings = model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Mine triplets
        triplets = mine_triplets(embeddings, labels, margin)
        
        if not triplets:
            continue
        
        # Compute loss
        batch_loss = None
        n_valid_triplets = 0
        
        for anchor_idx, pos_idx, neg_idx in triplets:
            anchor = embeddings[anchor_idx].unsqueeze(0)
            positive = embeddings[pos_idx].unsqueeze(0)
            negative = embeddings[neg_idx].unsqueeze(0)
            
            loss = F.triplet_margin_loss(anchor, positive, negative, margin=margin)
            if loss.item() > 0:  # Only count non-zero losses
                if batch_loss is None:
                    batch_loss = loss  # First valid loss becomes batch_loss
                else:
                    batch_loss += loss  # Add subsequent losses
                n_valid_triplets += 1
        
        if n_valid_triplets > 0:
            batch_loss /= n_valid_triplets
            
            # Backward and optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            
            running_loss += batch_loss.item()
            n_valid_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{batch_loss.item():.4f}" if n_valid_triplets > 0 else "N/A",
            'avg_loss': f"{(running_loss/n_valid_batches if n_valid_batches > 0 else 0):.4f}",
            'valid_triplets': n_valid_triplets
        })
    
    avg_loss = running_loss / max(n_valid_batches, 1)
    print(f"\nEpoch {epoch} completed. Average loss: {avg_loss:.4f}")
    print(f"Valid batches: {n_valid_batches}/{n_batches}")
    return avg_loss


def evaluate(model, val_loader, device):
    """
    Evaluate model on validation set
    
    Args:
        model: FaceNet model
        val_loader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        accuracy: Validation accuracy
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Get embeddings
            embeddings = model(images)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    # Concatenate all embeddings and labels
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Compute accuracy
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

    # Compute FAR at 10^-3
    far = compute_far(embeddings, labels, threshold=0.3)
    print(f"FAR @ 10^-3: {far:.6f}")

    return accuracy, far


# train and evaluation functions for ArcFace
def train_epoch_arcface(model, train_loader, criterion, optimizer, device, epoch, 
                       grad_accumulation_steps=2, scaler=None):
    """
    Memory-efficient training for one epoch using ArcFace loss with mixed precision
    
    Args:
        model: FaceNet model
        train_loader: DataLoader for training data
        criterion: ArcFace loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        grad_accumulation_steps: Number of steps to accumulate gradients
        scaler: GradScaler for mixed precision training
        
    Returns:
        avg_loss: Average loss for this epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Set smaller batch for display in progress bar
    n_batches = len(train_loader)
    progress_bar = tqdm(enumerate(train_loader), total=n_batches, 
                       desc=f"Epoch {epoch}")
    
    # Reset gradients at the beginning
    optimizer.zero_grad(set_to_none=True)
    
    # For gradient accumulation
    batch_count = 0
    accumulated_loss = 0
    
    for batch_idx, (images, labels) in progress_bar:
        images = images.to(device)
        labels = labels.to(device)
        
        # Mixed precision training
        if scaler is not None:
            # Forward pass with autocast
            with torch.amp.autocast('cuda'):
                embeddings = model(images)
                loss, logits = criterion(embeddings, labels)
                loss = loss / grad_accumulation_steps  # Scale loss
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            accumulated_loss += loss.item() * grad_accumulation_steps
            
            # Update weights after accumulating gradients
            batch_count += 1
            if batch_count % grad_accumulation_steps == 0:
                # This debug line may help identify the issue
                # print(f"Before step: {scaler._per_optimizer_states}")
                # scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Track metrics
                running_loss += accumulated_loss
                accumulated_loss = 0
        else:
            # Standard precision training (original code)
            embeddings = model(images)
            loss, logits = criterion(embeddings, labels)
            loss = loss / grad_accumulation_steps  # Scale loss
            
            # Backward pass
            loss.backward()
            accumulated_loss += loss.item() * grad_accumulation_steps
            
            # Update weights after accumulating gradients
            batch_count += 1
            if batch_count % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Track metrics
                running_loss += accumulated_loss
                accumulated_loss = 0
        
        # Compute accuracy
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item() * grad_accumulation_steps:.4f}",
            'avg_loss': f"{running_loss/(batch_idx+1):.4f}",
            'acc': f"{100.*correct/total:.2f}%"
        })
        
        # Clear cache periodically
        if batch_idx % 10 == 0 and hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
    
    # Handle the case when the last batch doesn't trigger an optimizer step
    if batch_count % grad_accumulation_steps != 0:
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = running_loss / n_batches
    accuracy = 100. * correct / total
    
    print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
    return avg_loss


def evaluate_arcface(model, val_loader, device):
    """
    Evaluate model on validation set when using ArcFace
    
    Args:
        model: FaceNet model
        val_loader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        accuracy: Validation accuracy
    """
    model.eval()
    embeddings_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            
            # Get embeddings
            embeddings = model(images)
            
            embeddings_list.append(embeddings)
            labels_list.append(labels)
    
    # Concatenate all embeddings and labels
    embeddings = torch.cat(embeddings_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    # Compute accuracy using nearest neighbor approach
    correct = 0
    total = 0
    
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
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
    
    # Compute FAR at 10^-3
    far = compute_far(embeddings, labels, threshold=0.3)
    print(f"FAR @ 10^-3: {far:.6f}")

    return accuracy, far