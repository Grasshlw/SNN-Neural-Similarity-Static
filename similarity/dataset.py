import os
import numpy as np


class NeuralDataset:
    def __init__(self, dataset_name, brain_areas, data_dir, **kwargs):
        self.dataset_name = dataset_name
        self.brain_areas = brain_areas
        self.data_dir = data_dir

        self.neural_dataset = {}
        for i in range(len(self.brain_areas)):
            self.neural_dataset[self.brain_areas[i]] = eval(f"self.{dataset_name}")(self.brain_areas[i], **kwargs)
    
    def allen_natural_scenes(self, brain_area, time_step, exclude, threshold, _mean_time_step=True):
        neural_data = np.load(os.path.join(self.data_dir, self.dataset_name, f"{brain_area}_{time_step}.npy"))

        if exclude:
            shr = np.load(os.path.join(self.data_dir, self.dataset_name, f"shr_{brain_area}.npy"))
            neural_data = neural_data[:, :, shr >= threshold]

        neural_data = neural_data / 50
        if _mean_time_step:
            neural_data = np.sum(neural_data, axis=1)
            neural_data /= 25

        return neural_data

    def macaque_face(self, brain_area, exclude, threshold, time_step=None, _mean_time_step=None):
        neural_data = np.load(os.path.join(self.data_dir, self.dataset_name, f"{brain_area}.npy"))

        if exclude:
            noise_ceiling = np.load(os.path.join(self.data_dir, self.dataset_name, "noise_ceiling.npy"))
            neural_data = neural_data[:, noise_ceiling >= threshold]
        
        return neural_data

    def macaque_synthetic(self, brain_area, time_step=None, exclude=None, threshold=None, _mean_time_step=None):
        return np.load(os.path.join(self.data_dir, self.dataset_name, f"{brain_area}.npy"))

    def __len__(self):
        return len(self.brain_areas)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.neural_dataset[self.brain_areas[key]]
        elif isinstance(key, str):
            return self.neural_dataset[key]
        else:
            raise KeyError(f"Unknown key: {key}")
    
    def keys(self):
        return self.brain_areas
