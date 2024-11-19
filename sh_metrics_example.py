# Import Spectral Hashing
from metrics.metrics_framework import MetricsFramework
from models.spectral_hashing import SH as SpectralHash
from metrics import feature_loader

import numpy as np

# Load the features from the CIFAR-10 dataset
fl = feature_loader.FeatureLoader("cifar-10")
cifar10_validation = fl.validation
features = []
for i in range(len(fl.training)):
    features.append(fl.training[i][1])

# Convert the features to a numpy array
features = np.array(features)
print("Features shape:", features.shape)

number_of_bits = 16

# Initialize Spectral Hashing with the features data
spectral_hash = SpectralHash.SpectralHashing(number_of_bits)
spectral_hash.fit(features)

# Select a query feature and reshapes so it works with SH
query_feature = features[0].reshape(1, -1)

# Query Spectral Hashing to find hash code for the query feature
query_hash_code = spectral_hash.query(query_feature)
print("Hash codes for query image:", query_hash_code)

metrics_framework = MetricsFramework(spectral_hash.query, fl.training)
metrics_framework.calculate_metrics(cifar10_validation, 999999)