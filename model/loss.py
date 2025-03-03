import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math

class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining"""
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        """
        Calculate triplet loss
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class as anchor)
            negative: Negative embeddings (different class from anchor)
            
        Returns:
            loss: Mean triplet loss
        """
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        
        # Triplet loss formula: max(pos_dist - neg_dist + margin, 0)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        return torch.mean(loss)

# def mine_triplets(embeddings, labels, mining_method='batch_hard', margin=0.2):
#     """
#     Mine triplets for training based on specified mining method
    
#     Args:
#         embeddings: Tensor of face embeddings
#         labels: Tensor of identity labels
#         mining_method: Method for mining triplets ('batch_hard' or 'batch_all')
#         margin: Margin for triplet loss
        
#     Returns:
#         triplets: List of tuples (anchor_idx, positive_idx, negative_idx)
#     """
#     if mining_method == 'batch_hard':
#         # Mine hardest triplets
#         triplets = []
        
#         for i in range(len(embeddings)):
#             anchor_label = labels[i]
#             anchor_emb = embeddings[i]
            
#             # Find positive samples (same label as anchor)
#             pos_indices = [j for j in range(len(labels)) if labels[j] == anchor_label and j != i]
#             if not pos_indices:
#                 continue
                
#             # Find negative samples (different label from anchor)
#             neg_indices = [j for j in range(len(labels)) if labels[j] != anchor_label]
#             if not neg_indices:
#                 continue
            
#             # Find hardest positive (furthest from anchor)
#             hardest_pos_idx = None
#             hardest_pos_dist = -1
            
#             for pos_idx in pos_indices:
#                 pos_dist = torch.norm(anchor_emb - embeddings[pos_idx], p=2)
#                 if pos_dist > hardest_pos_dist:
#                     hardest_pos_dist = pos_dist
#                     hardest_pos_idx = pos_idx
            
#             # Find hardest negative (closest to anchor)
#             hardest_neg_idx = None
#             hardest_neg_dist = float('inf')
            
#             for neg_idx in neg_indices:
#                 neg_dist = torch.norm(anchor_emb - embeddings[neg_idx], p=2)
#                 if neg_dist < hardest_neg_dist:
#                     hardest_neg_dist = neg_dist
#                     hardest_neg_idx = neg_idx
            
#             # Add hardest triplet
#             if hardest_pos_idx is not None and hardest_neg_idx is not None:
#                 triplets.append((i, hardest_pos_idx, hardest_neg_idx))
        
#         return triplets
#     else:
#         # Default to batch all triplets
#         triplets = []
#         for i in range(len(embeddings)):
#             anchor_label = labels[i]
#             pos_indices = [j for j in range(len(labels)) if labels[j] == anchor_label and j != i]
#             neg_indices = [j for j in range(len(labels)) if labels[j] != anchor_label]
            
#             if not pos_indices or not neg_indices:
#                 continue
                
#             for pos_idx in pos_indices:
#                 for neg_idx in neg_indices:
#                     triplets.append((i, pos_idx, neg_idx))
        
#         return triplets

def mine_triplets(embeddings, labels, margin=0.2):
    """
    Mine triplets with multiple fallback strategies to guarantee triplets are found
    """
    triplets = []
    
    for anchor_idx in range(len(embeddings)):
        anchor_label = labels[anchor_idx]
        anchor_emb = embeddings[anchor_idx]
        
        # Get all positives for this anchor
        pos_indices = [i for i in range(len(labels)) if labels[i] == anchor_label and i != anchor_idx]
        
        # Get all negatives for this anchor
        neg_indices = [i for i in range(len(labels)) if labels[i] != anchor_label]
        # print(f"Anchor {anchor_idx} has {len(pos_indices)} positives and {len(neg_indices)} negatives")
        
        if not pos_indices or not neg_indices:
            continue
            
        # STRATEGY 1: Try semi-hard negatives
        for pos_idx in pos_indices:
            pos_emb = embeddings[pos_idx]
            pos_dist = torch.norm(anchor_emb - pos_emb, p=2) ** 2
            
            semi_hard_found = False
            for neg_idx in neg_indices:
                neg_emb = embeddings[neg_idx]
                neg_dist = torch.norm(anchor_emb - neg_emb, p=2) ** 2
                
                # Semi-hard negative condition
                if pos_dist < neg_dist < pos_dist + margin:
                    triplets.append((anchor_idx, pos_idx, neg_idx))
                    semi_hard_found = True
                    break
            
            # If we found a semi-hard negative, move to next positive
            if semi_hard_found:
                # print(f"Found semi-hard negative for anchor {anchor_idx}")
                continue
                
            # STRATEGY 2: No semi-hard found, use the hardest negative (closest to anchor)
            neg_dists = [(neg_idx, torch.norm(anchor_emb - embeddings[neg_idx], p=2) ** 2) 
                        for neg_idx in neg_indices]
            closest_neg_idx = min(neg_dists, key=lambda x: x[1])[0]
            triplets.append((anchor_idx, pos_idx, closest_neg_idx))

    # print(f"Found {len(triplets)} triplets before strategy 3")
    # STRATEGY 3: If still no triplets, create triplets with all anchor-positive pairs and random negatives
    if not triplets:
        print("No triplets found with normal strategies, using random triplets")
        for anchor_idx in range(len(embeddings)):
            anchor_label = labels[anchor_idx]
            
            pos_indices = [i for i in range(len(labels)) if labels[i] == anchor_label and i != anchor_idx]
            neg_indices = [i for i in range(len(labels)) if labels[i] != anchor_label]
            
            if not pos_indices or not neg_indices:
                continue
                
            # Use random positives and negatives
            pos_idx = random.choice(pos_indices)
            neg_idx = random.choice(neg_indices)
            triplets.append((anchor_idx, pos_idx, neg_idx))
    
    return triplets


# ArcFace + UNPG
class ArcFaceLoss(nn.Module):
    """ArcFace loss with option to use UNPG following the repository implementation"""
    def __init__(self, embedding_size=512, num_classes=0, s=64.0, m=0.5, 
                 use_unpg=False, wisker_size=1.5):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.s = s  # scale factor
        self.m = m  # margin
        self.use_unpg = use_unpg
        self.wisker_size = wisker_size
        
        # Initialize class weight parameters
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)
        
        # For ArcFace calculation
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
        # Cross entropy loss
        self.cross_entropy = nn.CrossEntropyLoss()

    def filter_negative_pairs(self, similarities):
        """Filter using box-and-whisker algorithm as in the repository"""
        l = similarities.size(0)
        sorted_x = torch.sort(input=similarities, descending=False)[0]
        
        lower_quartile = sorted_x[int(0.25 * l)]
        upper_quartile = sorted_x[int(0.75 * l)]
                
        IQR = (upper_quartile - lower_quartile)        
        minimum = lower_quartile - self.wisker_size * IQR        
        maximum = upper_quartile + self.wisker_size * IQR
        mask = torch.logical_and(sorted_x <= maximum, sorted_x >= minimum)
        return sorted_x[mask]


    def convert_label_to_similarity(self, normed_feature, label):
        """Extract positive and negative similarity pairs from batch"""
        similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
        label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

        positive_matrix = label_matrix.triu(diagonal=1)
        negative_matrix = label_matrix.logical_not().triu(diagonal=1)

        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)
        return (similarity_matrix[positive_matrix], similarity_matrix[negative_matrix])


    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Compute ArcFace cosine similarity
        cosine = F.linear(embeddings, weights)
        
        # Apply ArcFace margin
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
                
        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, labels[index, None], self.m)
        
        # Apply arccosine then add margin
        cosine_with_margin = cosine.clone()
        cosine_with_margin.acos_()
        cosine_with_margin[index] += m_hot
        cosine_with_margin.cos_()
        
        if not self.use_unpg:
            # Just standard ArcFace
            return self.cross_entropy(self.s * cosine_with_margin, labels), cosine_with_margin
        
        # Get pairwise similarities for UNPG
        norm_x = F.normalize(embeddings)
        sp, sn = self.convert_label_to_similarity(norm_x, labels)
        
        # Apply box and whisker filtering to negative similarities
        if sn.numel() > 4:  # Ensure enough negatives for meaningful filtering
            sn_prime = self.filter_negative_pairs(sn)
            
            # Append filtered negative similarities to cosine matrix
            # Create a matrix that repeats sn_prime for each sample in batch
            batch_size = embeddings.size(0)
            one = torch.ones(batch_size, device=cosine.device).unsqueeze(1)
            aux_sn = one * sn_prime.unsqueeze(0)
            
            # Concatenate to the original cosine similarity matrix
            cosine_augmented = torch.cat([cosine_with_margin, aux_sn], dim=1)
            
            # Compute loss with augmented cosine matrix
            loss = self.cross_entropy(self.s * cosine_augmented, labels)
            
            return loss, cosine_with_margin
        
        # Fallback to standard ArcFace if not enough negatives
        return self.cross_entropy(self.s * cosine_with_margin, labels), cosine_with_margin