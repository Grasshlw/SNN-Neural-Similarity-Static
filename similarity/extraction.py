import os
import torch
import numpy as np
from tqdm import tqdm
import torchvision
import torchvision.models as models
import timm

from model.cornet import cornet_z, cornet_rt, cornet_s
from model.inception_resnet_v2 import inceptionresnetv2

from model.SEWResNet import *
from model.ShallowSEWResNet import *
from model.SpikingMobileNet import *
from model.RecurrentSEWResNet import *
from model.functional import *
from spikingjelly.activation_based import functional, neuron


class Extraction:
    def __init__(self, model, model_name, stimulus_path, device="cuda:0"):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
    
        self.features = None
        self.batch_size = 1
        self.set_stimulus(stimulus_path)

    def set_stimulus(self, stimulus_path):
        self.stimulus_path = stimulus_path
        self.stimulus = torch.load(stimulus_path)
        self.stimulus_change = True
    
    def build_dataloader(self):
        self.stimulus_dataset = torch.utils.data.TensorDataset(self.stimulus)
        self.n_stimulus = len(self.stimulus_dataset)
        self.stimulus_dataloader = torch.utils.data.DataLoader(self.stimulus_dataset, batch_size=self.batch_size)

    def replace_stimulus(self, replace_path, window):
        replace = torch.load(replace_path)
        stimulus = self.stimulus
        num_stimuli = stimulus.size(0)
        num_replace = int(np.ceil(num_stimuli / window))
        replace_index = np.random.randint(low=window, size=num_replace) + np.arange(0, num_replace * window, window)
        replace_index[-1] = min(replace_index[-1], num_stimuli - 1)
        stimulus[replace_index] = replace[replace_index]
        self.stimulus = stimulus
        self.stimulus_change = True
        return replace_index

    def hook_fn(self, module, inputs, outputs):
        pass

    def layer_extraction(self, layer_name, layer_dims):
        pass


class CNNStaticExtraction(Extraction):
    def __init__(self, model_name, stimulus_path, checkpoint_path=None, model_zoo=None, device="cuda:0"):
        if model_zoo == "torchvision":
            model = eval(f"models.{model_name}(pretrained=True)")
        else:
            model = eval(model_name)(checkpoint_path=checkpoint_path)
        
        super().__init__(model, model_name, stimulus_path, device)
        self._input = False
    
    def hook_fn(self, module, inputs, outputs):
        if self._input:
            self.features = inputs[0].data.cpu()
        else:
            self.features = outputs.data.cpu()

    def layer_extraction(self, layer_name, layer_dims, n_layer=None):
        if self.stimulus_change:
            self.build_dataloader()
            self.stimulus_change = False
        extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)
        self.model.eval()
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                if n_layer is None:
                    self.model(inputs)
                    extraction[n: n + bs] = self.features
                else:
                    outputs = self.model(inputs, n_layer)
                    extraction[n: n + bs] = outputs.data.cpu()
                n += bs
            hook.remove()

        return extraction


class ViTStaticExtraction(Extraction):
    def __init__(self, model_name, stimulus_path, model_zoo=None, device="cuda:0"):
        if model_zoo == "torchvision":
            model = eval(f"models.{model_name}(pretrained=True)")
        elif model_zoo == "timm":
            model = timm.create_model(model_name, pretrained=True)
        
        super().__init__(model, model_name, stimulus_path, device)
    
    def hook_fn(self, module, inputs, outputs):
        if isinstance(outputs, tuple):
            self.features = outputs[1].data.cpu()
        else:
            self.features = outputs.data.cpu()

    def layer_extraction(self, layer_name, layer_dims):
        if self.stimulus_change:
            self.build_dataloader()
            self.stimulus_change = False
        extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)
        self.model.eval()
        with torch.inference_mode():
            if layer_name == "self":
                hook = self.model.register_forward_hook(self.hook_fn)
            else:
                hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                self.model(inputs)
                extraction[n: n + bs] = self.features
                n += bs
            hook.remove()

        return extraction


class SNNStaticExtraction(Extraction):
    def __init__(self, model_name, stimulus_path, checkpoint_path=None, T=4, num_classes=1000, device="cuda:0"):
        model = eval(f"{model_name}")(cnf="ADD", num_classes=num_classes)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        set_step_mode(model, 'm', (ConvRecurrentContainer, ))
        set_backend(model, 'cupy', neuron.BaseNode, (ConvRecurrentContainer, ))

        super().__init__(model, model_name, stimulus_path, device)
        self.T = T
        self._mean = False
    
    def hook_fn(self, module, inputs, outputs):
        self.features.append(outputs.data.cpu())

    def layer_extraction(self, layer_name, layer_dims):
        if self.stimulus_change:
            self.build_dataloader()
            self.stimulus_change = False
        if self._mean:
            extraction = torch.zeros([self.n_stimulus] + layer_dims, dtype=torch.float)
        else:
            extraction = torch.zeros([self.n_stimulus] + [self.T] + layer_dims, dtype=torch.float)
        
        self.model.eval()
        functional.reset_net(self.model)
        with torch.inference_mode():
            hook = eval(f"self.model.{layer_name}").register_forward_hook(self.hook_fn)
            n = 0
            for inputs in tqdm(self.stimulus_dataloader):
                inputs = inputs[0].to(self.device)
                bs = len(inputs)
                inputs = inputs.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)

                self.features = []
                self.model(inputs)
                if len(self.features) == 1:
                    features = self.features[0]
                else:
                    features = torch.empty([self.T, bs] + layer_dims, dtype=torch.float)
                    for i in range(self.T):
                        features[i] = self.features[i]
                if self._mean:
                    extraction[n: n + bs] = features.mean(dim=0)
                else:
                    extraction[n: n + bs] = features.transpose(0, 1)
                functional.reset_net(self.model)
                n += bs
            hook.remove()

        return extraction
