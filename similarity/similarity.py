import os
import json
import torch
import numpy as np

from dataset import NeuralDataset
from visualmodel import VisualModel
from metric import CCAMetric, RSAMetric, RegMetric
from benchmark import Benchmark
from extraction import CNNStaticExtraction, ViTStaticExtraction, SNNStaticExtraction
from misc import *


def preset_neural_dataset(args):
    time_step = None
    exclude = False
    threshold = None
    if args.neural_dataset == "allen_natural_scenes":
        brain_areas = ['visp', 'visl', 'visrl', 'visal', 'vispm', 'visam']
        time_step = 8
        exclude = True
        threshold = 0.8
    elif args.neural_dataset == "macaque_face":
        brain_areas = ['AM']
        exclude = True
        threshold = 0.1
    elif args.neural_dataset == "macaque_synthetic":
        brain_areas = ['V4', 'IT']
    
    neural_dataset = NeuralDataset(
        dataset_name=args.neural_dataset,
        brain_areas=brain_areas,
        data_dir=args.neural_dataset_dir,
        time_step=time_step,
        exclude=exclude,
        threshold=threshold
    )
    return neural_dataset, exclude, threshold


def build_extraction(args):
    stimulus_name = f"stimulus_{args.neural_dataset}_"
    if args.model in ['inception_v3', 'inceptionresnetv2']:
        stimulus_name += "299.pt"
    else:
        stimulus_name += "224.pt"

    T = 0
    if args.model in cnn_list:
        model_zoo = None
        if args.model not in ['inceptionresnetv2', 'cornet_z', 'cornet_rt', 'cornet_s']:
            model_zoo = "torchvision"
        extraction = CNNStaticExtraction(
            model_name=args.model,
            stimulus_path=os.path.join(args.stimulus_dir, stimulus_name),
            checkpoint_path=args.checkpoint_path,
            model_zoo=model_zoo,
            device=args.device
        )
    elif args.model in vit_list:
        if args.model in vit_list[:4]:
            model_zoo = "torchvision"
        else:
            model_zoo = "timm"
        extraction = ViTStaticExtraction(
            model_name=args.model,
            stimulus_path=os.path.join(args.stimulus_dir, stimulus_name),
            model_zoo=model_zoo,
            device=args.device
        )
    elif args.model in snn_list:
        T = 4
        extraction = SNNStaticExtraction(
            model_name=args.model,
            stimulus_path=os.path.join(args.stimulus_dir, stimulus_name),
            checkpoint_path=args.checkpoint_path,
            device=args.device
        )
    return extraction, T


def preset_metric(args):
    if args.metric == "SVCCA":
        metric = CCAMetric(neural_reduction=(args.neural_dataset == "allen_natural_scenes"))
    elif args.metric == "TSVD-Regression":
        metric = RegMetric()
    elif args.metric == "RSA":
        metric = RSAMetric()
    return metric


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="Neural Representation Similarity for Static Stimuli")
    
    parser.add_argument("--model", default="sew_resnet18", type=str, help="name of model")
    parser.add_argument("--checkpoint-path", default=None, type=str, help="path of pretrained model checkpoint (default is None, which means that the checkpoint is provided by torchvision or timm)")

    parser.add_argument("--neural-dataset", default="allen_natural_scenes", type=str, choices=["allen_natural_scenes", "macaque_face", "macaque_synthetic"], help="name of neural dataset")
    parser.add_argument("--neural-dataset-dir", default="neural_dataset/", type=str, help="directory for storing neural dataset")

    parser.add_argument("--metric", default="SVCCA", type=str, choices=["SVCCA", "TSVD-Regression", "RSA"], help="name of similarity metric")

    parser.add_argument("--stimulus-dir", default="stimulus/", type=str, help="directory for stimulus")
    parser.add_argument("--device", default="cuda:0", type=str, help="device for extracting features")

    parser.add_argument("--output-dir", default="results/", help="directory to save results of representational similarity")

    args = parser.parse_args()
    return args


def main(args):
    extraction, args.T = build_extraction(args)
    with open(f"model_layers/{args.model}.json", 'r') as f:
        layers_info = json.load(f)
    layers_info = layers_info[:-1]
    visual_model = VisualModel(
        model_name=args.model,
        layers_info=layers_info,
        extraction=extraction,
        _time_step=args.T,
        _mean_time_step=True,
        _normalize=True
    )

    neural_dataset, args.exclude, args.threshold = preset_neural_dataset(args)
    metric = preset_metric(args)
    save_dir = os.path.join(args.output_dir, args.metric, args.neural_dataset)
    benchmark = Benchmark(
        neural_dataset=neural_dataset,
        metric=metric,
        save_dir=save_dir
    )
    print(args)
    benchmark(visual_model)


if __name__ == "__main__":
    args = get_args()
    main(args)