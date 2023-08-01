import os
import time
import numpy as np


class Benchmark:
    def __init__(self, neural_dataset, metric, save_dir=None):
        self.neural_dataset = neural_dataset
        self.metric = metric
        self.save_dir = save_dir

        self.brain_areas = self.neural_dataset.keys()
        self.num_areas = len(self.neural_dataset)

        self.model_name = None
        self.num_layers = 0
    
    def _preset_print(self):
        self.area_length = 0
        for brain_area in self.brain_areas:
            if len(brain_area) > self.area_length: self.area_length = len(brain_area)
        
        self.layer_length = 1
        num_layers = self.num_layers
        while num_layers // 10 > 0:
            self.layer_length += 1
            num_layers //= 10
        self.layer_length += len(self.model_name) + 1

    def _print_score(self, brain_area, model_layer, score, _time):
        print(f"%-{self.area_length}s %-{self.layer_length}s: %-9.6f  time: %.4fs" % (brain_area, model_layer, score, _time))

    def _print_all_scores(self, scores):
        print("All scores:")

        print(f"%{self.layer_length}s" % (' '), end='')
        for brain_area in self.brain_areas:
            print(f" %-11s" % (brain_area), end='')
        print()

        for i in range(scores.shape[0]):
            print(f"%-{self.layer_length}s" % (f"{self.model_name}_{i + 1}"), end='')
            for j in range(scores.shape[1]):
                print(f" %-11.8f" % (scores[i, j]), end='')
            print()

    def _save_scores(self, scores):
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, f"{self.model_name}.npy"), scores)

    def __call__(self, visual_model):
        self.model_name = visual_model.model_name
        self.num_layers = len(visual_model)

        self._preset_print()
        scores = np.zeros((self.num_layers, self.num_areas))
        for layer_index in range(self.num_layers):
            print()
            print(f"Extracting features of layer {layer_index + 1} of {self.model_name}")
            model_data = visual_model[layer_index]
            for area_index, brain_area in enumerate(self.brain_areas):
                start_time = time.time()
                neural_data = self.neural_dataset[area_index]
                scores[layer_index, area_index] = self.metric.score(model_data, neural_data)
                self._print_score(brain_area, f"{self.model_name}_{layer_index + 1}", scores[layer_index, area_index], time.time() - start_time)
        print()       
        self._print_all_scores(scores)
        
        if self.save_dir is not None:
            self._save_scores(scores)
