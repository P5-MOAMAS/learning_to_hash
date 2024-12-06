# Import Spectral Hashing
import numpy as np

from models.spectral_hashing import SH as SpectralHash
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

    results = []
    for i in range(len(encode_len)):
        # Initialize Spectral Hashing with the features data
        spectral_hash = SpectralHash.SpectralHashing(encode_len[i])
        spectral_hash.fit(features)

        metrics_framework = MetricsFramework(spectral_hash.query, fl, query_size)
        metrics_framework.calculate_metrics(dataset_name + "/SH_" + str(encode_len[i]) + "_bits_" + dataset_name, k)
