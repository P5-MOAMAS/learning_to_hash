from sympy import false

from models.DSH.train import *
from metrics import metrics_framework
from metrics import feature_loader
import torch
import torchvision.models as models

from models.lsh.Lsh_v2 import query_image

fl = feature_loader.FeatureLoader("cifar-10")
cifar10_validation = fl.validation


test_pair_dataset = PairDataset(f'C:/Users/Sandra/PycharmProjects/learning_to_hash/models/DSH/data_out', 'cifar', train=False)
test_image, _ = test_pair_dataset[0][:2]

model_path = f'C:/Users/Sandra/PycharmProjects/learning_to_hash/models/DSH/best_model.pth'
#model = torch.load(model_path, weights_only=True)
#model.eval()

model = LiuDSH(code_size=16, channels=3, size=32, num_classes=10).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
model.query_image(test_image)

print("Hash Code for the test image:", model.query_image(test_image))

metrics_framework.calculate_metrics(model.query_image, cifar10_validation, False)