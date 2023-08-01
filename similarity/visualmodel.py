import os
import torch
import numpy as np

from misc import *


class VisualModel:
    def __init__(self, model_name, layers_info, extraction, _time_step=0, _mean_time_step=False, _flatten_time_step=False, _normalize=False):
        self.model_name = model_name
        self.layers_info = layers_info
        self.extraction = extraction

        self._time_step = _time_step
        self._mean_time_step = _mean_time_step
        self._flatten_time_step = _flatten_time_step
        self._normalize = _normalize
    
    def _z_score(self, x):
        _mean = np.mean(x)
        _std = np.std(x)
        x = (x - _mean) / (_std + 1e-10)
        
        return x

    def _process_model_data(self, x):
        x = x.numpy()
        if self._time_step > 0:
            if self._mean_time_step:
                x = np.mean(x, axis=1)
                x = x.reshape((x.shape[0], -1))
            else:
                x = x.reshape((x.shape[0], x.shape[1], -1))
                x = x.transpose(0, 2, 1)
                if self._flatten_time_step:
                    x = x.reshape((x.shape[0], -1))
        else:
            x = x.reshape((x.shape[0], -1))
        if self._normalize:
            x = self._z_score(x)
        return x

    def __len__(self):
        return len(self.layers_info)
    
    def __getitem__(self, key):
        layer_name = self.layers_info[key][0]
        layer_dims = self.layers_info[key][1]
        if self.model_name in cnn_list:
            n_layer = None
            if self.model_name == "cornet_rt":
                n_layer = key + 1
            self.extraction._input = False
            if layer_dims == [-1]:
                layer_dims = [self.layers_info[key - 1][1][0]]
                self.extraction._input = True
        
            model_data = self.extraction.layer_extraction(layer_name, layer_dims, n_layer)
        else:
            model_data = self.extraction.layer_extraction(layer_name, layer_dims)
        model_data = self._process_model_data(model_data)
        return model_data
