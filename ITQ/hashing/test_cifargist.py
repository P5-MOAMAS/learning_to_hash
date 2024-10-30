import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from itq import ITQ
from itq import ITQ
from dataset import load_cifar10_gist
from evaluation import mean_average_precision, precision_recall

# Load CIFAR-10 features
root = './datasets/cifar10_gist512'
database_features, database_labels, query_features, query_labels = load_cifar10_gist(root)
database_features = database_features.astype('float32')
query_features = query_features.astype('float32')
n_train = database_features.shape[0]

# Check data shapes
print(f'Database shape: {database_features.shape}, Labels shape: {database_labels.shape}')
print(f'Query shape: {query_features.shape}, Labels shape: {query_labels.shape}')

scaler = StandardScaler(with_mean=True, with_std=True)
Database_features = scaler.fit_transform(database_features)
query_features = scaler.transform(query_features)

# Initialize ITQ
encode_len = 128
model = ITQ(encode_len)
model.fit(database_features)

# encode
query_b = model.encode(query_features)
database_b = model.encode(database_features)

# Evaluation
mAP = mean_average_precision(query_b, query_labels, database_b, database_labels)
precision, recall = precision_recall(query_b, query_labels, database_b, database_labels)

# Output results
print(f'encode_len = {encode_len}')
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)