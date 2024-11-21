from metrics.feature_loader import FeatureLoader

from metrics.metrics_framework import MetricsFramework
from models.ITQ.hashing.itq_model import *

# Load the features from the CIFAR-10 dataset
fl = FeatureLoader("cifar-10", False)
data = fl.training

features = [feature for (_, feature, _) in data]
k = 100

encode_len = [16]
results = []
for i in range(len(encode_len)):
    model = ITQ(encode_len[i])
    model.fit(features)

    metrics_framework = MetricsFramework(model.encode_single, data, 1000)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", round(results[i], 3))