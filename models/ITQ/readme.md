Python implementations of hashing model
---------------------------------------
### this project is using a modified implementation from https://github.com/wen-zhi/hashing 

The implemented hashing models include:
- PCA-ITQ

Requirements
------------

Make sure to install the following Python packages before running the code:

```bash
pip install numpy torch scikit-learn
```

Usage
------

The current implemented models are `ITQ with PCA`.

> You can download **MNIST-GIST512D** from [here](https://drive.google.com/open?id=14MG9OGekFROlbe-aLHuNlRMiFZITHFhh).

## ITQ

```python
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from itq import ITQ  # Ensure ITQ class is defined as provided below
from evaluation import mean_average_precision, precision_recall  # Evaluation functions
from dataset import load_cifar10_deep  # Dataset loading function

# Load CIFAR-10 Features
root = './features/cifar10_features/'
query_data, train_data, database_data = load_cifar10_deep(root)

query, query_label = query_data
train, train_label = train_data
database, database_label = database_data

print(f'Query shape: {query.shape}, Labels shape: {query_label.shape}')
print(f'Train shape: {train.shape}, Labels shape: {train_label.shape}')
print(f'Database shape: {database.shape}, Labels shape: {database_label.shape}')

# Initialize ITQ with desired encoding length
encode_len = 32
model = ITQ(encode_len)
model.fit(database)

# Encode the query and database features
query_b = model.encode(query)
database_b = model.encode(database)

# Evaluate the binary codes
mAP = mean_average_precision(query_b, query_label, database_b, database_label)
precision, recall = precision_recall(query_b, query_label, database_b, database_label)

# Display results
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)
```

## Methods
1. Iterative Quantization (ITQ)
2. The core ITQ method performs the following:

Dimensionality Reduction: Reduces feature dimensionality using PCA to the desired length (encode_len).
Binary Encoding: Encodes features to binary by iteratively optimizing the rotation matrix to minimize quantization loss.

# References

[1] M. S. Charikar, “**Similarity Estimation Techniques from Rounding Algorithms**,” in Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, New York, NY, USA, 2002, pp. 380–388, doi: 10.1145/509907.509965.  
[2] Y. Gong and S. Lazebnik, “**Iterative quantization: A procrustean approach to learning binary codes**,” in 2011 IEEE Conference on Computer Vision and Pattern Recognition, Piscataway, NJ, USA, Jun. 2011, pp. 817–824.  
