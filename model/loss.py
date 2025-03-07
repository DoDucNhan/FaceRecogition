import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ArcFace + UNPG
class ArcFaceLoss(nn.Module):
    """
    ArcFace loss with option to use UNPG (Utilizing Negative Pairs with Geometric meaning).
    
    This implementation includes numerical stability improvements and follows patterns
    from established implementations.
    
    Args:
        embedding_size: Dimension of the feature embeddings
        num_classes: Number of identity classes
        s: Scale factor for logits (typically 64.0)
        m: Angular margin in radians (typically 0.5)
        use_unpg: Whether to use the UNPG extension
        wisker_size: Box-and-whisker IQR multiplier for filtering negative pairs
        easy_margin: Whether to use the easy margin approximation for numerical stability
    """
    def __init__(self, embedding_size=512, num_classes=0, s=64.0, m=0.5, 
                 use_unpg=False, wisker_size=1.5, easy_margin=False):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s  # scale factor
        self.m = m  # margin
        self.use_unpg = use_unpg
        self.wisker_size = wisker_size
        self.easy_margin = easy_margin
        
        # Initialize class weight parameters
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-calculate constants for ArcFace
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)  # Decision boundary for cosine values
        self.safe_margin = self.sin_m * m  # Safe margin for numerical stability
        
        # Cross entropy loss
        self.cross_entropy = nn.CrossEntropyLoss()
        
        # Epsilon for numerical stability
        self.eps = 1e-7


    def filter_negative_pairs(self, similarities):
        """
        Filter negative pairs using the box-and-whisker algorithm.
        
        Args:
            similarities: Tensor of similarity values for negative pairs
            
        Returns:
            Filtered tensor of similarity values within acceptable range
        """
        if similarities.numel() < 4:
            return similarities  # Not enough samples for meaningful quartiles
            
        l = similarities.size(0)
        sorted_x = torch.sort(input=similarities, descending=False)[0]
        
        # Calculate quartiles
        lower_quartile = sorted_x[int(0.25 * l)]
        upper_quartile = sorted_x[int(0.75 * l)]
                
        # Apply IQR filtering
        IQR = (upper_quartile - lower_quartile)        
        minimum = lower_quartile - self.wisker_size * IQR        
        maximum = upper_quartile + self.wisker_size * IQR
        mask = torch.logical_and(sorted_x <= maximum, sorted_x >= minimum)
        return sorted_x[mask]


    def convert_label_to_similarity(self, normed_feature, label):
        """
        Extract positive and negative similarity pairs from feature batch.
        
        Args:
            normed_feature: Normalized feature embeddings
            label: Class labels for the batch
            
        Returns:
            Tuple of (positive_similarities, negative_similarities)
        """
        # Calculate pairwise similarities
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        
        # Create binary matrices for positive and negative pairs
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)
        positive_matrix = label_matrix.triu(diagonal=1)  # Upper triangle excluding diagonal
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        # Extract similarity values using masks
        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        
        # Return positive and negative similarities
        pos_sims = similarity_matrix[positive_matrix]
        neg_sims = similarity_matrix[negative_matrix]
        
        return (pos_sims, neg_sims)


    def forward(self, embeddings, labels):
        """
        Forward pass for ArcFace loss computation.
        
        Args:
            embeddings: Feature embeddings from the network
            labels: Ground truth class labels (-1 for ignored samples)
            
        Returns:
            loss: Computed loss value
            cosine_with_margin: Cosine similarities with margin applied (for debugging)
        """
        # Normalize embeddings and weights for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=1, eps=self.eps)
        weights = F.normalize(self.weight, p=2, dim=1, eps=self.eps)
        
        # Compute cosine similarity
        cosine = F.linear(embeddings, weights)
        
        # Find valid samples (labels != -1)
        valid_idx = torch.where(labels != -1)[0]
        
        # Apply ArcFace margin
        cosine_with_margin = cosine.clone()
        
        if len(valid_idx) > 0:
            target_cosine = cosine[valid_idx, labels[valid_idx].view(-1)]
            
            # Two approaches for adding margin:
            if self.easy_margin:
                # Easy margin approach (simpler but less accurate at boundaries)
                sin_theta = torch.sqrt(1.0 - torch.pow(target_cosine, 2))
                cos_theta_m = target_cosine * self.cos_m - sin_theta * self.sin_m
                
                # For samples near the decision boundary
                mask = target_cosine > self.threshold
                final_target_cosine = torch.where(
                    mask, 
                    cos_theta_m, 
                    target_cosine - self.safe_margin
                )
                
                # Update only valid indices with final cosine values
                cosine_with_margin[valid_idx, labels[valid_idx].view(-1)] = final_target_cosine
            else:
                # Original approach (more accurate) using arccosine
                one_hot = torch.zeros_like(cosine)
                one_hot.scatter_(1, labels.view(-1, 1), 1)
                
                # Create margin matrix for valid indices
                m_hot = torch.zeros(valid_idx.size(0), cosine.size(1), device=cosine.device)
                m_hot.scatter_(1, labels[valid_idx].view(-1, 1), self.m)
                
                # Apply arccosine, add margin, then apply cosine
                cosine_with_margin.acos_()
                cosine_with_margin[valid_idx] += m_hot
                cosine_with_margin.cos_()
        
        # Standard ArcFace without UNPG
        if not self.use_unpg:
            return self.cross_entropy(self.s * cosine_with_margin, labels), cosine_with_margin
        
        # UNPG extension - utilize negative pairs
        norm_x = F.normalize(embeddings, p=2, dim=1, eps=self.eps)
        sp, sn = self.convert_label_to_similarity(norm_x, labels)
        
        # Apply box-and-whisker filtering to negative similarities if enough pairs exist
        if sn.numel() > 4:
            sn_prime = self.filter_negative_pairs(sn)
            
            # Ensure we have at least some negative pairs after filtering
            if sn_prime.numel() > 0:
                # Create a matrix that repeats filtered negatives for each sample in batch
                batch_size = embeddings.size(0)
                one = torch.ones(batch_size, device=cosine.device).unsqueeze(1)
                aux_sn = one * sn_prime.unsqueeze(0)
                
                # Concatenate to the original cosine similarity matrix
                cosine_augmented = torch.cat([cosine_with_margin, aux_sn], dim=1)
                
                # Compute loss with augmented cosine matrix
                loss = self.cross_entropy(self.s * cosine_augmented, labels)
                
                return loss, cosine_with_margin
        
        # Fallback to standard ArcFace if not enough negatives or filtering removed all pairs
        return self.cross_entropy(self.s * cosine_with_margin, labels), cosine_with_margin
    

class TripletLoss(nn.Module):
    """
    Triplet loss with semi-hard triplet mining following the original FaceNet paper
    
    Args:
        margin: Margin for triplet loss (default: 0.2)
        squared: Use squared Euclidean distance (default: True)
    """
    def __init__(self, margin=0.2, squared=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Tensor of shape (batch_size, embedding_size)
            labels: Tensor of shape (batch_size)
            
        Returns:
            triplet_loss: Scalar tensor containing the triplet loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get pairwise distance matrix
        pairwise_dist = self._pairwise_distances(embeddings)
        
        # For each anchor, get the hardest positive and semi-hardest negative
        hardest_positive_dist = self._get_hardest_positive_dist(pairwise_dist, labels)
        semi_hard_negative_dist = self._get_semi_hard_negative_dist(pairwise_dist, labels, hardest_positive_dist)
        
        # Calculate triplet loss
        loss = F.relu(hardest_positive_dist - semi_hard_negative_dist + self.margin)
        
        # Count number of non-zero (active) triplets
        non_zero_triplets = torch.sum(loss > 1e-16).float()
        
        # Return mean over positive triplets
        return torch.mean(loss), embeddings
    
    def _pairwise_distances(self, embeddings):
        """Compute pairwise distances between embeddings"""
        # Get dot product (batch_size, batch_size)
        dot_product = torch.matmul(embeddings, embeddings.t())
        
        # Get squared L2 norm for each embedding
        square_norm = torch.diagonal(dot_product)
        
        # Calculate pairwise distances
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * <a, b>
        distances = square_norm.unsqueeze(0) + square_norm.unsqueeze(1) - 2.0 * dot_product
        
        # Because of computation errors, some distances might be negative
        # So we eliminate these values by setting them to zero
        distances = F.relu(distances)
        
        if not self.squared:
            # Add small epsilon for numerical stability before taking square root
            mask = distances > 0
            distances = distances + 1e-16 * (1.0 - mask.float())
            distances = torch.sqrt(distances)
        
        return distances
    
    def _get_hardest_positive_dist(self, pairwise_dist, labels):
        """
        For each anchor, get the hardest positive (largest distance to same class)
        
        Args:
            pairwise_dist: Pairwise distance matrix
            labels: Labels for each sample
            
        Returns:
            hardest_positive_dist: Distances of hardest positives
        """
        # Create mask for positive pairs (same class)
        mask_positives = labels.unsqueeze(0) == labels.unsqueeze(1)
        
        # Remove diagonal elements (same sample)
        mask_positives = mask_positives.logical_xor(torch.eye(labels.size(0), device=labels.device).bool())
        
        # Replace non-positives with very large number
        masked_dists = pairwise_dist.clone()
        masked_dists[~mask_positives] = 1e12
        
        # Get hardest positive (maximum distance)
        hardest_positive_dist = torch.min(masked_dists, dim=1)[0]
        
        return hardest_positive_dist
    
    def _get_semi_hard_negative_dist(self, pairwise_dist, labels, hardest_positive_dist):
        """
        For each anchor, get the semi-hard negative (different class, but not too far)
        
        Args:
            pairwise_dist: Pairwise distance matrix
            labels: Labels for each sample
            hardest_positive_dist: Distances of hardest positives
            
        Returns:
            semi_hard_negative_dist: Distances of semi-hard negatives
        """
        # Create mask for negative pairs (different class)
        mask_negatives = labels.unsqueeze(0) != labels.unsqueeze(1)
        
        # Create semi-hard negative mask (negatives that are closer than hardest positive + margin)
        semi_hard_mask = (pairwise_dist < hardest_positive_dist.unsqueeze(1) + self.margin) & mask_negatives
        
        # If there are no semi-hard negatives, use the hardest negatives instead
        # First, identify samples that have no semi-hard negatives
        no_semi_hard = (torch.sum(semi_hard_mask.float(), dim=1) == 0)
        
        # Initialize array to store selected negative distances
        selected_negative_dist = torch.zeros_like(hardest_positive_dist)
        
        # For each anchor...
        for i in range(labels.size(0)):
            if no_semi_hard[i]:
                # No semi-hard negative, use hardest negative (closest)
                masked_dists = pairwise_dist[i].clone()
                masked_dists[~mask_negatives[i]] = 1e12
                selected_negative_dist[i] = torch.min(masked_dists)
            else:
                # Has semi-hard negatives, choose closest semi-hard
                masked_dists = pairwise_dist[i].clone()
                masked_dists[~semi_hard_mask[i]] = 1e12
                selected_negative_dist[i] = torch.min(masked_dists)
        
        return selected_negative_dist