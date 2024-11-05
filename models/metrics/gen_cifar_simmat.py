import sys

class CifarSimilarityMatrix:

    @staticmethod
    def load():
        import pickle
        with open("models/metrics/sim_matrix", 'rb') as fo:
            matrix = pickle.load(fo, encoding='bytes')

        return CifarSimilarityMatrix(matrix)


    @staticmethod
    def create_matrix():
        sim_matrix = []

        for _ in range(10):
            sim_matrix.append([])
        
        for i, labels in enumerate(load_cifar_labels()):
            for x, label in enumerate(labels):
                sim_matrix[label].append(x + i * 10000)

        return CifarSimilarityMatrix(sim_matrix)

    def save(self):
        import pickle
        with open("models/metrics/sim_matrix", 'wb') as file:
            pickle.dump(self.matrix, file)

    def __init__(self, matrix: list[list[int]]):
        self.matrix = matrix

    def get_related(self, x: int) -> list[int]:
        for lst in self.matrix:
            for idx in lst:
                if idx == x:
                    return lst

        return []


def load_cifar_labels():
    fp = "cifar-10-batches-py/"

    fileNames = []
    for i in range(1, 6):
        fileNames.append(fp + "data_batch_" + str(i))

    for name in fileNames:
        yield unpickle(name)[b'labels']


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == "__main__":
    mat1 = CifarSimilarityMatrix.create_matrix()
    
    print(mat1.get_related(int(sys.argv[1])))
