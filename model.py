import torch
import torch.nn as nn
import numpy as np

class OpenDlign(nn.Module):
    def __init__(self, clip_model, tokenizer):
        super().__init__()
        self.logit_scale = clip_model.logit_scale
        
        self.clip_model = clip_model
        self.tokenizer = tokenizer

    def encode_image(self, image, state):
        return self.clip_model.encode_image(image, state = state)
        
    def forward(self, rgb, depth):
        feature_A_all = []
        feature_B_all = []
        
        for i in range(rgb.shape[0]):
            depth_features = self.clip_model.encode_image(depth[i], state = 'depth_branch')
            depth_features = depth_features / depth_features.norm(dim=-1, keepdim=True)
            depth_features = depth_features.mean(dim=0)
            depth_features = depth_features / depth_features.norm(dim=-1, keepdim=True)
            feature_A_all.append(depth_features)

            rgb_features = self.clip_model.encode_image(rgb[i], state = 'rgb_branch')
            rgb_features = rgb_features / rgb_features.norm(dim=-1, keepdim=True)
            rgb_features= rgb_features.mean(dim=0)
            rgb_features = rgb_features / rgb_features.norm(dim=-1, keepdim=True)
            feature_B_all.append(rgb_features)

        feature_A_all = torch.stack(feature_A_all)
        feature_B_all = torch.stack(feature_B_all)
        
        return {'feature_A': feature_A_all, 'feature_B': feature_B_all, 'logit_scale': self.logit_scale.exp()}
