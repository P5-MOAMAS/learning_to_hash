# Import Spectral Hashing
import numpy as np

from models.spectral_hashing import SH as SpectralHash
from utility.feature_loader import FeatureLoader
from utility.metrics_framework import calculate_encoding_time, save_results, MetricsFramework

# The traditional methods use pre extracted features.
# The encode times are an average over all datasets (Encoding times was within 50ms) using 1000 single queries for each.
# Preprocessing times varied between datasets, so they are in an array [mnist, cifar, nuswide]
gpu_encode = 1.244
cpu_encode = 14.130
preprocessing = [1.102, 1.068, 1.801]

################################################################
############################ Config ############################
################################################################
k = [100, 250, 500, 1000, 2500, 5000]
query_size = 10000
encode_len = [8, 16, 32, 64]
dataset_names = ["mnist", "cifar-10", "nuswide"]

for i, dataset_name in enumerate(dataset_names):
    fl = FeatureLoader(dataset_name)
    features = np.asarray(fl.data[query_size:])

    encoding_times = {}

    results = []
    for encode_length in encode_len:
        # Initialize Spectral Hashing with the features data
        spectral_hash = SpectralHash.SpectralHashing(encode_length)
        spectral_hash.fit(features)

        metrics_framework = MetricsFramework(spectral_hash.query, fl, query_size)
        metrics_framework.calculate_metrics(dataset_name + "/SH_" + str(encode_length) + "_bits_" + dataset_name, k)
        gpu_mean, cpu_mean, _ = calculate_encoding_time(encode_length, fl, query_cpu=spectral_hash.query)
        bit_times = {"gpu": gpu_mean, "cpu": cpu_mean, "processing": 0, "total_gpu": cpu_mean + gpu_encode + preprocessing[i],
                     "total_cpu": cpu_mean + cpu_encode + preprocessing[i]}
        encoding_times[encode_length] = bit_times
    save_results(encoding_times, dataset_name + "/SH_encoding_times")
