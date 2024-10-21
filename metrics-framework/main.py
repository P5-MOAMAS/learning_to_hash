import random

from torch import nn
from feature_loader import FeatureLoader
from label_loader import LabelLoader
import torch
import gc

"""
The class picks a random subset from the dataset specified

self.model: 
    The model being evaluated.

self.sample:
    The dataset the model is evaluated on.

self.switch_sample:
    Should switch to a new dataset. This includes setting self.sample and self.labels
    This functions should also use random to pick a random subset of the data.
    Maybe use del to make sure there is no memory living longer than needed.

self.query_model:
    Should call the model given the features
    This function should make sure the model is in the correct 'state',
    such as model.eval and pytorch.nograd
"""

class ModelEvaluator:
    def __init__(self, dataset_name: str, model: nn.Module):
        self.model = model
        self.sample = None
        self.labels = None
        self.switch_sample(dataset_name)
        pass

    def calculate_recall(self):
        pass

    def calculate_precision(self):
        pass

    def query_model(self, features: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(features)
        return pred

    """
    Separate function in case sample should be selected across all batches
    """
    def get_sample(self, sample_size: int, dataset_name: str, batch_id: int = 1):
        data = FeatureLoader(dataset_name)[batch_id]
        labels = LabelLoader(dataset_name)[batch_id]

        selected_ids = random.sample(range(len(data)), sample_size)
        selected_data = [data[x] for x in selected_ids]
        selected_labels = [labels[x] for x in selected_ids]

        del data, labels
        gc.collect()

        return selected_data, selected_labels


    def switch_sample(self, dataset_name: str, sample_size: int = 1000):
        self.sample, self.labels = self.get_sample(sample_size, dataset_name)