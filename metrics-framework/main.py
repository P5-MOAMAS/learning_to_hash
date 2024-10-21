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
    def __init__(self, sample):
        self.model = None
        self.switch_sample(sample)
        pass

    def calculate_recall(self):
        pass

    def calculate_precision(self):
        pass

    def query_model(self, features):
        pass

    def switch_sample(self, sample: str):
        self.sample = None
        self.labels = None
        pass
