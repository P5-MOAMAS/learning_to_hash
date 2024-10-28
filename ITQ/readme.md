Python implementations of hashing model
---------------------------------------

The implemented hashing models include:
- LSH<sup>1</sup>
- PCA-ITQ, CCA-ITQ<sup>2</sup>
- SDH<sup>3</sup>

Requirements
------------

```
# create a new env
conda create --name hashing scikit-learn numpy
# activate the new env
conda activate hashing
```

Usage
------

The current implemented models are `LSH, ITQ, ITQ_CCA, SDH`. All the models share the same interface,
so it is very easy to use them. The Following are two examples:

> You can download **MNIST-GIST512D** from [here](https://drive.google.com/open?id=14MG9OGekFROlbe-aLHuNlRMiFZITHFhh).

## ITQ

```python
from hashing.model import ITQ
from hashing.dataset import load_mnist_gist
from hashing.evaluation import mean_average_precision, precision_recall

# load mnist-512d data
root = './datasets/mnist-gist-512d.npz'
query_data, train_data, database_data = load_mnist_gist(root)
query, query_label = query_data
train, _ = train_data
database, database_label = database_data

# ITQ
encode_len = 32
model = ITQ(encode_len)
model.fit(database)

# encode
query_b = model.encode(query)
database_b = model.encode(database)

# evaluation
mAP = mean_average_precision(query_b, query_label,
                             database_b, database_label)
precision, recall = precision_recall(query_b, query_label,
                                     database_b, database_label)
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)
```

## ITQ-CCA

```python
from hashing.model import ITQ_CCA
from hashing.dataset import load_mnist_gist
from hashing.evaluation import mean_average_precision, precision_recall
from hashing.utils import one_hot_encoding

# load mnist-512d data
root = './datasets/mnist-gist-512d.npz'
query_data, train_data, database_data = load_mnist_gist(root)
query, query_label = query_data
train, train_label = train_data
database, database_label = database_data

# ITQ
model = ITQ_CCA(encode_len=32)
train_label = one_hot_encoding(train_label, 10)
model.fit(train, train_label)

# encode
query_b = model.encode(query)
database_b = model.encode(database)

# evaluation
mAP = mean_average_precision(query_b, query_label,
                             database_b, database_label)
precision, recall = precision_recall(query_b, query_label,
                                     database_b, database_label)
print(f'mAP = {mAP}')
print("Precision:")
print(precision)
print("Recall:")
print(recall)
```

# References

[1] M. S. Charikar, “**Similarity Estimation Techniques from Rounding Algorithms**,” in Proceedings of the Thiry-Fourth Annual ACM Symposium on Theory of Computing, New York, NY, USA, 2002, pp. 380–388, doi: 10.1145/509907.509965.  
[2] Y. Gong and S. Lazebnik, “**Iterative quantization: A procrustean approach to learning binary codes**,” in 2011 IEEE Conference on Computer Vision and Pattern Recognition, Piscataway, NJ, USA, Jun. 2011, pp. 817–824.  
[3] F. Shen, C. Shen, W. Liu, and H. Tao Shen, “**Supervised Discrete Hashing**,” in 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Piscataway, NJ, USA, Jun. 2015, pp. 37–45.
