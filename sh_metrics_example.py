# Import Spectral Hashing
from metrics.metrics_framework import MetricsFramework
from models.spectral_hashing import SH as SpectralHash
from metrics.feature_loader import FeatureLoader

import numpy as np

# Load the features from the CIFAR-10 dataset
fl = FeatureLoader("cifar-10", False)
data = fl.training

# Flatten the images for compatibility with LSH (each image as a 1D feature vector)
features = [feature for (_, feature, _) in data]

# Convert the features to a numpy array
features = np.array(features)
print("Features shape:", features.shape)

k = 100
encode_len = [16]
results = []
for i in range(len(encode_len)):
    # Initialize Spectral Hashing with the features data
    spectral_hash = SpectralHash.SpectralHashing(encode_len[i])
    spectral_hash.fit(features)

    metrics_framework = MetricsFramework(spectral_hash.query, data, 1000)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", round(results[i], 3))