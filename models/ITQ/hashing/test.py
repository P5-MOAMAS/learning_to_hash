import numpy as np
import torch
from itq import ITQ
from dataset import load_cifar10_deep
from evaluation import mean_average_precision, precision_recall

# Load CIFAR-10 features
root = './deep_features/'
query_data, database_data = load_cifar10_deep(root)

query, query_label = query_data
database, database_label = database_data

# Check data shapes
print(f'Query shape: {query.shape}, Labels shape: {query_label.shape}')
print(f'Database shape: {database.shape}, Labels shape: {database_label.shape}')

# Initialize ITQ
encode_len = 32
model = ITQ(encode_len)
model.fit(database)
model.save_model('itq_model.pth')

# Load the model
loaded_model = ITQ(encode_len)  # Reinitialize the model
loaded_model.load_model('itq_model.pth')  # Load the saved model

# Encode the query and database
query_b = loaded_model.encode(query)
database_b = loaded_model.encode(database)

# Evaluation
mAP = mean_average_precision(query_b, query_label, database_b, database_label)
precision, recall = precision_recall(query_b, query_label, database_b, database_label)

# Output results
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)
