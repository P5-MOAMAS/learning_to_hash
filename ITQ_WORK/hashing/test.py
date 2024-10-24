from itq import ITQ, ITQ_CCA
from dataset import load_mnist_gist
from evaluation import mean_average_precision, precision_recall

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