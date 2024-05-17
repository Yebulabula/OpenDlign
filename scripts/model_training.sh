#!/bin/bash

python train.py --root point_cloud_dataset/ --clip_model ViT-H-14-quickgelu --output_checkpoint_dir checkpoints --eval_dataset modelnet40 --log_dir logging --config-file model_configs/ViT-H-14-quickgelu.yaml