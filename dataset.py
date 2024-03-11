import torch
from torch.utils.data import Dataset
import os 
import numpy as np  
from PIL import Image


class ZeroShotDataset(Dataset):
    def __init__(self, depth_files, labels, transform = None, classnames=None):
        self.depth = depth_files
        self.transform = transform 
        self.labels = labels
        
        label_dict = {category:i for i, category in enumerate(classnames)}
        self.labels = [label_dict[category.replace('_',' ')] for category in self.labels]
    
    def __len__(self):
        return len(self.depth)
    
    def __getitem__(self, idx):
        depth = np.load(self.depth[idx])
        if self.transform:
            depth = torch.stack([self.transform(torch.from_numpy(d).permute(2, 0, 1)) for d in depth], dim = 0)

        return depth, self.labels[idx]

class AlignDataset(Dataset):
    def __init__(self, RGB_files, depth_files, transform = None, tokenizer=None):
        self.RGB = RGB_files
        self.depth = depth_files
        self.transform = transform 
        self.tokenizer = tokenizer
        
        self.imgs_list = ['{}.png'.format(i) for i in range(10)]
    
    def __len__(self):
        return len(self.depth)
    
    def __getitem__(self, idx):
        depth = np.load(self.depth[idx])
        if self.transform:
            depth = torch.stack([self.transform(torch.from_numpy(d).permute(2, 0, 1)) for d in depth], dim = 0)
            rgb = [np.array(Image.open(os.path.join(self.RGB[idx], img)).convert('RGB'), dtype=np.float32) for img in self.imgs_list]
            rgb = torch.stack([self.transform(torch.from_numpy(r).permute(2, 0, 1)) for r in rgb], dim = 0)
            
        return rgb, depth