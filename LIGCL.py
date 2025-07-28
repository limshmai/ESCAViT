import torch
import torch.nn as nn
import torch.nn.functional as F

class LIGCL(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, original_embeddings, mixup_embeddings, original_labels, lam):

        original_embeddings = F.normalize(original_embeddings, dim=1) 
        mixup_embeddings = F.normalize(mixup_embeddings, dim=1)       
        
        batch_size = original_embeddings.shape[0]
        
        similarity = torch.matmul(mixup_embeddings, original_embeddings.T) 
        similarity = similarity / self.temperature

        logits_max, _ = torch.max(similarity, dim=1, keepdim=True)
        logits = similarity - logits_max.detach()

        labels = original_labels.contiguous().view(-1, 1) 

        pos_mask = (labels == labels.T).float() 

        neg_mask = (labels != labels.T).float()

        pos_mask = pos_mask * lam.view(-1, 1)    

        exp_logits = torch.exp(logits) * neg_mask

        denom = exp_logits.sum(dim=1, keepdim=True) + 1e-6
        log_prob = logits - torch.log(denom)

        numerator = (pos_mask * log_prob).sum(dim=1) 
        pos_count = pos_mask.sum(dim=1) + 1e-6        
        mean_log_prob_pos = numerator / pos_count

        loss = -mean_log_prob_pos.mean()
        
        return loss
