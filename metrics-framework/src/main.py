import random

from torch import nn
from feature_loader import FeatureLoader
from gen_cifar_simmat import CifarSimilarityMatrix
from label_loader import LabelLoader
from hash_lookup import pre_gen_hash_codes

import torch
import gc

def calculate_recall(dataset_name: str, id: int, image_ids: list[int]):
    sim_matrix = None
    match dataset_name:
        case "cifar-10":
            try:
                sim_matrix = CifarSimilarityMatrix.load()
            except:
                CifarSimilarityMatrix.create_matrix().save()
                sim_matrix = CifarSimilarityMatrix.load()
        case _:
            raise LookupError("No similarity matrix can be found for: " + dataset_name)

    related = sim_matrix.get_related(id)
    count = sum([1 if x in related else 0 for x in image_ids])

    if len(related) == 0:
        return 0
    else:
        return count / len(related)


def calculate_precision(dataset_name: str, id: int, image_ids: list[int]):
    sim_matrix = None
    match dataset_name:
        case "cifar-10":
            try:
                sim_matrix = CifarSimilarityMatrix.load()
            except:
                CifarSimilarityMatrix.create_matrix().save()
                sim_matrix = CifarSimilarityMatrix.load()
        case _:
            raise LookupError("No similarity matrix can be found for: " + dataset_name)

    related = sim_matrix.get_related(id)
    count = sum([1 if x in related else 0 for x in image_ids])
    
    if len(image_ids) == 0:
        return 0
    else:
        return count / len(image_ids)

"""
The class picks a random subset from the dataset specified

self.model: 
    The model being evaluated.

self.sample:
    The dataset the model is evaluated on.

self.switch_sample:
    Should switch to a new dataset. This includes setting self.sample and self.labels
    This functions should also use random to pick a random subset of the data.
    Maybe use del to make sure there is no memory living longer than needed.

self.query_model:
    Should call the model given the features
    This function should make sure the model is in the correct 'state',
    such as model.eval and pytorch.nograd
"""
class ModelEvaluator:
    def __init__(self, dataset_name: str, model: nn.Module):
        self.database = None
        self.model = model
        self.sample = None
        self.labels = None
        self.dataset_name = dataset_name
        self.switch_sample(dataset_name)

    """
    id is the image identity
    image_idxs is the list of all image identities retrieved
    This function calculates the recall, meaning of only related images
    retrieved what percentage do these make up of all related images.
    """
    def calculate_recall(self, id: int, image_ids: list[int]):
        return calculate_recall(self.dataset_name, id, image_ids)


    """
    id is the image identity
    image_idxs is the list of all image identities retrieved
    This function calculates the precision, meaning of the images
    retrieved what percentage of these are related.
    """
    def calculate_precision(self, id: int, image_ids: list[int]):
        return calculate_recall(self.dataset_name, id, image_ids)


    def calculate_metrics_for_id(self, id: int):
        self.prepare_dataset()
        neighbors = self.database.get_neighbors(id)
        ids = []
        for neighbor in neighbors:
            ids.extend(self.database.get_images(neighbor))

        print("Recall: ", calculate_recall(self.dataset_name, id, ids))
        print("Precision: ", calculate_precision(self.dataset_name, id, ids))


    """
    Pre-gens all hash codes for the current dataset
    """
    def prepare_dataset(self):
        feature_loader = FeatureLoader(self.dataset_name)
        for batch in feature_loader:
            self.database = pre_gen_hash_codes(self.query_model, batch, self.database)


    def query_model(self, features: torch.Tensor):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(features)
        return pred

    """
    Separate function in case sample should be selected across all batches
    """
    def get_sample(self, sample_size: int, batch_id: int = 1):
        data = FeatureLoader(self.dataset_name)[batch_id]
        labels = LabelLoader(self.dataset_name)[batch_id]

        selected_ids = random.sample(range(len(data)), sample_size)
        selected_data = [data[x] for x in selected_ids]
        selected_labels = [labels[x] for x in selected_ids]

        del data, labels
        gc.collect()

        return selected_data, selected_labels


    def switch_sample(self, dataset_name: str, sample_size: int = 1000):
        self.sample, self.labels = self.get_sample(sample_size, dataset_name)


if __name__ == '__main__':
    true_positives = [0, 19, 22, 23, 25, 72, 95, 103, 104, 117, 124, 125, 132, 143, 151, 154, 164, 187, 200, 204, 209, 210, 224, 228, 231, 232, 234, 235, 242, 243, 245, 248, 249, 286, 292, 298, 313, 326, 327, 347, 350, 351, 355, 361, 368, 387, 409, 437, 451, 452, 464, 473, 488, 525, 529, 532, 552, 556, 571, 587, 588, 591, 619, 620, 625, 633, 640, 645, 651, 655, 667, 680, 682, 692, 710, 718, 720, 721, 728, 738, 745, 755, 770, 781, 807, 818, 819, 836, 837, 838, 854, 862, 863, 896, 899, 903, 914, 920, 921, 923, 929, 931, 935, 937, 960, 961, 985, 1013, 1016, 1017, 1023, 1027, 1031, 1035, 1038, 1047, 1051, 1053, 1063, 1085, 1101, 1126, 1131, 1132, 1154, 1155, 1159, 1161, 1175, 1176, 1191, 1192, 1194, 1210, 1220, 1222, 1228, 1237, 1246, 1248, 1259, 1317, 1318, 1327, 1342, 1360, 1362, 1367, 1371, 1374, 1393, 1402, 1403, 1420, 1436, 1447, 1453, 1484, 1485, 1529, 1531, 1536, 1537, 1550, 1552, 1563, 1579, 1584, 1585, 1596, 1613, 1627, 1628, 1635, 1637, 1642, 1646, 1678, 1687, 1690, 1722, 1730, 1733, 1741, 1743, 1760, 1761, 1766, 1802, 1807, 1809, 1819, 1836, 1837, 1842, 1874, 1876, 1917, 1933, 1944, 1948, 1955, 1956, 1959, 1978, 1982, 1989, 2019, 2040, 2043, 2050, 2099, 2100, 2111, 2112, 2115, 2124, 2134, 2146, 2150, 2152, 2159, 2179, 2203, 2222, 2226, 2239, 2241, 2252, 2267, 2274, 2276, 2288, 2295, 2296, 2299, 2310, 2336, 2356, 2364, 2369, 2400, 2407, 2409, 2411, 2424, 2439, 2475, 2479, 2494, 2503, 2510, 2517, 2519, 2536, 2542, 2556, 2566, 2571, 2583, 2596, 2602, 2605, 2607, 2618, 2623, 2625, 2631, 2638, 2650, 2653, 2674, 2681, 2684, 2690, 2694, 2697, 2713, 2729, 2751, 2774, 2777, 2786, 2796, 2810, 2822, 2831, 2836, 2866, 2871, 2872, 2874, 2888, 2893, 2898, 2909, 2915, 2924, 2925, 2937, 2939, 2941, 2955, 2965, 2968, 2970, 2972, 2978, 2989, 2992, 2997, 2998, 3004, 3010, 3018, 3033, 3041, 3048, 3070, 3091, 3095, 3096, 3098, 3106, 3111, 3122, 3139, 3154, 3168, 3192, 3200, 3203, 3207, 3223, 3227, 3241, 3243, 3262, 3266, 3290, 3307, 3311, 3323, 3341, 3369, 3401, 3403, 3411, 3412, 3435, 3437, 3439, 3458, 3470, 3500, 3504, 3521, 3525, 3528, 3530, 3532, 3554, 3563, 3567, 3588, 3589, 3590, 3593, 3603, 3621, 3631, 3636, 3655, 3656, 3676, 3682, 3683, 3686, 3698, 3707, 3710, 3712, 3716, 3724, 3729, 3730, 3739, 3742, 3754, 3775, 3780, 3783, 3804, 3805, 3809, 3823, 3825, 3826, 3833, 3841, 3849, 3861, 3865, 3873, 3892, 3912, 3926, 3942, 3943, 3947, 3976, 3986, 3991, 3994, 4015, 4022, 4026, 4031, 4035, 4062, 4064, 4077, 4082, 4086, 4094, 4095, 4105, 4118, 4123, 4126, 4130, 4143, 4144, 4147, 4148, 4152, 4167, 4181, 4190, 4194, 4201, 4208, 4217, 4234, 4235, 4267, 4276, 4278, 4279, 4284, 4309, 4324, 4377, 4389, 4391, 4403, 4404, 4409, 4415, 4417, 4421, 4430, 4434, 4440, 4450, 4452, 4488, 4546, 4550, 4568, 4570, 4590, 4596, 4605, 4606, 4637, 4644, 4659, 4663, 4668, 4678, 4682, 4685, 4693, 4731, 4734, 4751, 4752, 4766, 4767, 4770, 4785, 4786, 4789, 4792, 4805, 4810, 4813, 4824, 4831, 4843, 4870, 4875, 4877, 4881, 4899, 4904, 4907, 4911, 4926, 4937, 4946, 4947, 4948, 4951, 4959, 4966, 4980, 4999, 5000, 5019, 5020, 5022, 5025, 5032, 5041, 5055, 5059, 5066, 5081, 5085, 5095, 5109, 5122, 5124, 5134, 5142, 5148, 5159, 5167, 5169, 5176, 5177, 5182, 5206, 5223, 5230, 5263, 5268, 5291, 5292, 5302, 5308, 5318, 5338, 5353, 5355, 5359, 5361, 5374, 5375, 5376, 5380, 5381, 5398, 5402, 5409, 5419, 5424, 5429, 5435, 5443, 5456, 5474, 5489, 5497]
    false_positives = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    idx = true_positives + false_positives

    print("Recall:")
    print(calculate_recall("cifar-10", 0, idx))

    print("Precision:")
    
    print(calculate_precision("cifar-10", 0, idx))
