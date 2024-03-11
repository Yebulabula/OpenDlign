'''
 * Copyright (c) 2023, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Le Xue
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils
        
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None

    def forward(self, outputs):
        feature_A = outputs['feature_A']  
        feature_B = outputs['feature_B']

        local_batch_size = feature_A.size(0)
        logit_scale = outputs['logit_scale']
        
        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=feature_A.device
            )
            self.last_local_batch_size = local_batch_size
        
        feature_A = F.normalize(feature_A, dim=-1, p=2)
        feature_B = F.normalize(feature_B, dim=-1, p=2)

        feature_A_all, feature_B_all = utils.all_gather_batch([feature_A, feature_B])   

        logits_per_A_B = logit_scale * feature_A @ feature_B_all.t()
        logits_per_B_A = logit_scale * feature_B @ feature_A_all.t()

        loss_A_B = (F.cross_entropy(logits_per_A_B, self.labels) + \
                    F.cross_entropy(logits_per_B_A, self.labels))/2
        
        with torch.no_grad():
            pred = torch.argmax(logits_per_A_B, dim=-1)
            correct = pred.eq(self.labels).sum()
            A_B_acc = correct / local_batch_size
            
            pred = torch.argmax(logits_per_B_A, dim=-1)
            correct = pred.eq(self.labels).sum()
            B_A_acc = correct / local_batch_size

        return {'loss_A_B': loss_A_B, 'A_B_acc': A_B_acc, 'B_A_acc': B_A_acc} 
    
if __name__ == "__main__":
    citerion = ContrastiveLoss()
    outputs = {
        'pc_embed': torch.rand(2, 256),
        'text_embed': torch.rand(2, 256),
        'image_embed': torch.rand(2, 256),
        'logit_scale': 1.0
    }