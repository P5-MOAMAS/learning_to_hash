from metrics import metrics_framework
from metrics import feature_loader
from models.ITQ.hashing.itq_model import *

# Load the features from the CIFAR-10 dataset
fl = feature_loader.FeatureLoader("cifar-10")
cifar10_validation = fl.validation

features = []
for i in range(len(fl.training)):
    features.append(fl.training[i][1])
del fl

encode_len = 8
model = ITQ(encode_len)
model.fit(features)

query_image = features[0]
features = np.delete(features, 0, axis=0)
print("Removed the query image from the dataset.")
query_image = query_image.reshape(-1,1)

database_b = model.encode(features)
query_hash_codes = model.encode_single(query_image)
print("Hash codes for query image:", query_hash_codes)

metrics_framework.calculate_metrics(model.encode_single, cifar10_validation, False)