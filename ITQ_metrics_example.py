from metrics import feature_loader

from metrics.metrics_framework import MetricsFramework
from models.ITQ.hashing.itq_model import *

# Load the features from the CIFAR-10 dataset
fl = feature_loader.FeatureLoader("cifar-10")
data = fl.validation

features = [feature for (_, feature, _) in data]
k = 9000

encode_len = [2, 4, 8, 16, 32, 64]
results = []
for i in range(len(encode_len)):
    model = ITQ(encode_len[i])
    model.fit(features)

    metrics_framework = MetricsFramework(model.encode_single, data, 2000)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", round(results[i], 3))