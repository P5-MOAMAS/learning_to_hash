import numpy as np

from models.lsh import Lsh
from utility.feature_loader import FeatureLoader
from utility.metrics_framework import MetricsFramework

################################################################
############################ Config ############################
################################################################
k = [100, 250, 500, 1000, 2500, 5000]
query_size = 10000
encode_len = [8, 16, 32, 64]
dataset_names = ["mnist", "cifar-10", "nuswide"]

for dataset_name in dataset_names:
    fl = FeatureLoader(dataset_name)
    features = np.asarray(fl.data[query_size:])

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
        metrics_framework.calculate_metrics(dataset_name + "/LSH_" + str(encode_len[i]) + "_bits_" + dataset_name, k)
