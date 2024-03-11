#!/bin/bash

python zero_shot.py --root /mnt/new_drive/YeMao_Dataset/point_cloud_dataset/ --clip_model ViT-H-14-quickgelu --checkpoint_path model_checkpoint/ViT-H-14-quickgelu.pth --eval_dataset modelnet40 --log_dir logging