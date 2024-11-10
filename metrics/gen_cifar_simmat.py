class SimilarityMatrix:
    @staticmethod
    def create_matrix(dataset: list[tuple[int, list, int]]):
        sim_matrix = {}
        for index, feature, label in dataset:
            if label in sim_matrix:
                sim_matrix[label].append(index)
            else:
                sim_matrix[label] = [index]

        return SimilarityMatrix(sim_matrix)

    def __init__(self, matrix: dict[int, list[int]]):
        self.matrix = matrix

    def get_related(self, label: int) -> list[int]:
        if label in self.matrix:
            return self.matrix[label]
        else:
            return []
