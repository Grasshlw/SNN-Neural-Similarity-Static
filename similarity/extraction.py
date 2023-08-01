import os
import torch
from tqdm import tqdm
import torchvision
import torchvision.models as models
import timm

from model.cornet import cornet_z, cornet_rt, cornet_s
from model.inception_resnet_v2 import inceptionresnetv2

from model.SEWResNet import *
from model.ShallowSEWResNet import *
from model.SpikingMobileNet import *
from spikingjelly.activation_based import functional, neuron


class Extraction:
    def __init__(self, model, model_name, stimulus_path, device="cuda:0"):
        self.model = model.to(device)
        self.model_name = model_name
        self.set_stimulus(stimulus_path)
        self.device = device
    
        self.features = None
        self.batch_size = 1
        self.build_dataloader()

    def set_stimulus(self, stimulus_path):
        self.stimulus_path = stimulus_path
        self.stimulus = torch.load(stimulus_path)
    
    def build_dataloader(self):
        self.stimulus_dataset = torch.utils.data.TensorDataset(self.stimulus)
        self.n_stimulus = len(self.stimulus_dataset)
        self.stimulus_dataloader = torch.utils.data.DataLoader(self.stimulus_dataset, batch_size=self.batch_size)

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
    def __init__(self, model_name, stimulus_path, checkpoint_path=None, device="cuda:0"):
        model = eval(f"{model_name}")(cnf="ADD", num_classes=1000)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

        functional.set_step_mode(model, 'm')
        functional.set_backend(model, 'cupy', neuron.BaseNode)

        super().__init__(model, model_name, stimulus_path, device)
        self.T = 4
        self._mean = False
    
    def hook_fn(self, module, inputs, outputs):
        self.features = outputs.data.cpu()

    def layer_extraction(self, layer_name, layer_dims):
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
                self.model(inputs)
                if self._mean:
                    extraction[n: n + bs] = self.features.mean(dim=0)
                else:
                    extraction[n: n + bs] = self.features.transpose(0, 1)
                functional.reset_net(self.model)
                n += bs
            hook.remove()

        return extraction
