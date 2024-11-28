# Import Spectral Hashing
from utility.metrics_framework import MetricsFramework
from models.spectral_hashing import SH as SpectralHash
from utility.feature_loader import FeatureLoader

import numpy as np

# Load the features from the CIFAR-10 dataset
fl = FeatureLoader("cifar-10")

k = 500
query_size = 10000
features = np.asarray(fl.data[query_size:])

encode_len = [8, 16, 32, 64]
results = []
for i in range(len(encode_len)):
    # Initialize Spectral Hashing with the features data
    spectral_hash = SpectralHash.SpectralHashing(encode_len[i])
    spectral_hash.fit(features)

    metrics_framework = MetricsFramework(spectral_hash.query, fl.data, fl.labels, query_size)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", results[i])