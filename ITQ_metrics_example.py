from models.ITQ.hashing.itq_model import *
from utility.feature_loader import FeatureLoader
from utility.metrics_framework import MetricsFramework

# Load the features from the CIFAR-10 dataset
fl = FeatureLoader("cifar-10")

k = 500
query_size = 1000

encode_len = [8]
results = []
for i in range(len(encode_len)):
    print("---------------- Calculating metrics for bit length", encode_len[i], "----------------")
    model = ITQ(encode_len[i])
    model.fit(fl.data[query_size:])

    metrics_framework = MetricsFramework(model.encode, fl, query_size, multi_encoder=True)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", results[i])
