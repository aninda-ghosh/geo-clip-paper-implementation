import torch
import torch.nn as nn

class Contrastive_Loss(nn.Module):
    def __init__(self):
        super(Contrastive_Loss, self).__init__()

    def forward(self, cosine_sim_matrix):
        logits = cosine_sim_matrix

        exp_logits = torch.exp(logits)        
            
        diag_logits = torch.diag(exp_logits)

        #get the sum of the exponential of the logits
        exp_logits_sum = exp_logits.sum(1)

        #compute the loss
        loss = -torch.log(diag_logits / exp_logits_sum)

        #compute the mean loss
        loss = loss.mean()

        return loss