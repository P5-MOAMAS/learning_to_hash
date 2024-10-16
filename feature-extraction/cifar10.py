import pickle
import sys

"""
Loaded in this way, each of the batch files contains a dictionary with the following elements:
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.

    labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

The dataset contains another file, called batches.meta. It too contains a Python dictionary object. It has the following entries:
    label_names -- a 10-element list which gives meaningful names to the numeric labels in the labels array described above. For example, label_names[0] == "airplane", label_names[1] == "automobile", etc.
"""

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar10(batch_number):
    if batch_number >= 6:
        print("Batch number does not exist!")
        sys.exit(1)
    elif batch_number <= 0:
        print("Batch number does not exist!")
        sys.exit(1)

    with open("cifar-10-batches-py/data_batch_" + str(batch_number), 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict[b'data']

