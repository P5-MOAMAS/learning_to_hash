import numpy as np

from models.lsh import Lsh
from utility.feature_loader import FeatureLoader
from utility.metrics_framework import MetricsFramework

# Load the features from the CIFAR-10 dataset
fl = FeatureLoader("cifar-10")

k = 500
query_size = 1000
features = np.asarray(fl.data[query_size:])

encode_len = [8]
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

    metrics_framework = MetricsFramework(image_lsh.query, fl, query_size)
    mAP = metrics_framework.calculate_metrics(k)
    results.append(mAP)

print("---------------- Result using k value of", k, "----------------")
for i in range(len(results)):
    print(encode_len[i], "bit length result:", results[i])
