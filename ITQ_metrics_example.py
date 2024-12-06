from models.ITQ.hashing.itq_model import *
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

    for i in range(len(encode_len)):
        print("---------------- Calculating metrics for bit length", encode_len[i], "----------------")
        model = ITQ(encode_len[i])
        model.fit(fl.data[query_size:])

        metrics_framework = MetricsFramework(model.encode, fl, query_size, multi_encoder=True)
        metrics_framework.calculate_metrics(dataset_name + "/ITQ_" + str(encode_len[i]) + "_bits_" + dataset_name, k)
