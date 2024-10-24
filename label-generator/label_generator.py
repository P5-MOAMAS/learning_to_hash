import pickle
import numpy as np
from numpy.linalg import norm

from progressbar import progressbar


def load_labels(f, start: int, end: int):
    labels = []
    for i in range(start, end + 1):
        labels.extend(f(i))

    return labels


def load_cifar10(batch: int):
    with open("cifar-10-batches-py/data_batch_" + str(batch), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    labels = dict[b'labels']
    del dict
    return labels


def create_matrix_for_batch(labels: list, batch: int, batch_size: int):
    start = (batch - 1) * batch_size
    end = start + batch_size
    matrix = [[] for _ in range(start, end)]
    for i in progressbar(range(start, end)):
        for j in range(len(labels)):
            if labels[j] == labels[i]:
                matrix[i].append(1)
            else:
                matrix[i].append(0)


    return matrix


def create_similarity_matrix_sklearn(matrix: list, labels: list):
    return np.dot(matrix,labels)/(norm(matrix, axis=1)*norm(labels))


if __name__ == "__main__":
    labels = load_labels(load_cifar10, 1, 5)
    matrix = create_matrix_for_batch(labels, 1, 10000)
    similarity = create_similarity_matrix_sklearn(matrix, labels)
    print(similarity.shape)
