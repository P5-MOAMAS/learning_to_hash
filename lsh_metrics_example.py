import numpy as np

from metrics.feature_loader import FeatureLoader
from metrics.metrics_framework import MetricsFramework
from models.lsh import Lsh

# Load the features from the CIFAR-10 dataset
fl = FeatureLoader("cifar-10", False)
data = fl.training

# Flatten the images for compatibility with LSH (each image as a 1D feature vector)
features = [feature for (_, feature, _) in data]
features = np.array(features)

k = 100
encode_len = [16]
results = []
for i in range(len(encode_len)):
    # Set LSH parameters
    num_tables = 1
    num_bits_per_table = encode_len[i]
    pca_components = 50

    # Initialize LSH with the features data
    image_lsh = Lsh.LSH(
        features,
        num_tables=num_tables,
        num_bits_per_table=num_bits_per_table,
        pca_components=pca_components,
    )

    metrics_framework = MetricsFramework(image_lsh.query, data, 1000)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", round(results[i], 3))