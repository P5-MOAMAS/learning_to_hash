from sympy import false
from models.DSH.train import *
from metrics.feature_loader import FeatureLoader
from metrics.metrics_framework import MetricsFramework
from models.ITQ.hashing.itq_model import *
import torch
import torchvision.models as models
from models.lsh.Lsh_v2 import query_image

fl = FeatureLoader("cifar-10")
data = fl.training
k = 9000

test_pair_dataset = PairDataset('/models/DSH/data_out', 'cifar', train=False) #add path to where cifar-10-batches is located
test_image, _ = test_pair_dataset[0][:2]

model_path = '/models/DSH/best_model_cifar_16.pth' #add path to trained model
model = LiuDSH(code_size=16, channels=3, size=32, num_classes=10).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
model.query_image(test_image)

print("Hash Code for the test image:", model.query_image(test_image))
results = []
metrics_framework = MetricsFramework(model.query_image, data, 2000)
mAP = metrics_framework.calculate_metrics(k)
results.append(mAP)

print("Results:", results)