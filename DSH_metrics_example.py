from utility.data_loader import Dataloader
from models.DSH.train import *
from utility.metrics_framework import MetricsFramework
import torch

trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
fl = Dataloader("cifar-10")

k = 500
query_size = 10000

test_pair_dataset = PairDataset('./models/DSH/data_out', 'cifar', train=False) #add path to where cifar-10-batches is located
test_image, _ = test_pair_dataset[0][:2]

model_path = './models/DSH/best_model_cifar_16.pth' #add path to trained model
model = LiuDSH(code_size=16, channels=3, size=32, num_classes=10).to(device)
model.load_state_dict(torch.load(model_path, weights_only=False))
model.eval()
model.query_image(test_image)

print("Hash Code for the test image:", model.query_image(test_image))
results = []
metrics_framework = MetricsFramework(model.query_image, fl.data, fl.labels, query_size, trans=trans)
mAP = metrics_framework.calculate_metrics(k)
results.append(mAP)

print("Results for k =", k, ":", results)