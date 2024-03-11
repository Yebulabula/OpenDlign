# OpenDlign: Enhancing Open-World 3D Learning with Depth-Aligned Images
[Project Website](https://Yebulabula.github.io/OpenDlign/)

[***News***] [11/03/2024] OpenDlign pre-trained models and datasets have been released. 🔥🔥🔥

Official implementation of [OpenDlign: Enhancing Open-World 3D Learning with Depth-Aligned Images](https://arxiv.org/abs/12356)

![avatar](img/architecture.pdf)

# Instructions
OpenDlign is a multimodal framework for learning open-world 3D representations. It leverages depth-aligned images generated from point cloud-projected depth maps. Unlike CAD-rendered images, our generated images provide rich, realistic color and texture diversity while preserving geometric and semantic consistency with the depth maps. Our experiments demonstrate OpenDlign's superior performance in zero-shot and few-shot classification, 3D object detection, and cross-modal retrieval, especially with real-scanned 3D objects.

## Install environments
We pre-train OpenDlign on 1 Nvidia A100 GPU, the code is tested with CUDA==11.3 and pytorch==1.11.0\
```conda create -n OpenDlign python=3.8``` \
```conda activate OpenDlign``` \
```conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch``` \
```pip install -r requirements.txt```

## Datasets
The processed evaluation data (i.e., ModelNet40, ScanObjectNN, OmniObject3D) can be found [here](https://huggingface.co/datasets/OpenDlign/OpenDlign-Datasets)

## Pretrained Models
The pre-trained OpenDlign models, which are integrated with various CLIP variants (e.g., ViT-H-14, ViT-L-14, ViT-B-16, ViT-B-32), are available [here](https://huggingface.co/OpenDlign/OpenDlign-Models)

## Inference
Update the root path of your downloaded evaluation dataset before running the following command:
```bash scripts/zero_shot.sh```

## Training
You are free
Update the root path of your downloaded training dataset before running the following command:
```bash scripts/model_training.sh```

## Citation

If you find our code is helpful, please cite our paper:

```
@article{ye2023opendlign,
  title={OpenDlign: Enhancing Open-World 3D Learning with Depth-Aligned Images},
  author={Ye Mao and Junpeng Jing and Krystian Mikolajczyk},
  year={2024}
}
