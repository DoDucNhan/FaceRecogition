import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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