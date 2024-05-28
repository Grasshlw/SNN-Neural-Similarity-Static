# SNN-Neural-Similarity-Static

Official implementation of "[Deep Spiking Neural Networks with High Representation Similarity Model Visual Pathways of Macaque and Mouse](https://doi.org/10.1609/aaai.v37i1.25073)" (**AAAI2023 Oral**).

By Liwei Huang, Zhengyu Ma, Liutao Yu, Huihui Zhou, Yonghong Tian.

We are the first to apply deep SNNs to fit neural representations and shed light on visual processing mechanisms in both macaques and mice, demonstrating the potential of SNNs as a novel and powerful tool for research on the visual system.

![overview](./imgs/overview.PNG)

## Requirements

In order to run this project you will need:

- Python3
- PyTorch
- SpikingJelly
- The following packages: numpy, tqdm, scikit-learn

## SNN Training

The code is stored in the file folder `train`. It supports single GPU or multiple GPUs.

Train on the ImageNet:

```
python train_imagenet.py --epochs 320 --batch-size 32 --opt sgd --lr 0.1 --lr-scheduler cosa --lr-warmup-epochs 5 --lr-warmup-decay 0.01 --amp --model-name sew_resnet18 --T 4 --output-path logs/
```

## Representational Similarity

The code is stored in the file folder `similarity`.

Normal experiment:

```
python similarity.py --model sew_resnet18 --train-dataset imagenet --checkpoint-path model_checkpoint/imagenet/sew_resnet18.pth --neural-dataset allen_natural_scenes --neural-dataset-dir neural_dataset/ --metric SVCCA --stimulus-dir stimulus/ --output-dir results/
```

## Citation

If you find our work is useful for your research, please kindly cite our paper:

```
@article{
    Huang_Ma_Yu_Zhou_Tian_2023,
    title={Deep Spiking Neural Networks with High Representation Similarity Model Visual Pathways of Macaque and Mouse},
    volume={37},
    url={https://ojs.aaai.org/index.php/AAAI/article/view/25073},
    DOI={10.1609/aaai.v37i1.25073},
    number={1},
    journal={Proceedings of the AAAI Conference on Artificial Intelligence},
    author={Huang, Liwei and Ma, Zhengyu and Yu, Liutao and Zhou, Huihui and Tian, Yonghong},
    year={2023},
    month={Jun.},
    pages={31-39}
}
```