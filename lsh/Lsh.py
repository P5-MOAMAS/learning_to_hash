import argparse
import sys
from os import listdir
from os.path import isfile, join
from typing import Dict, List, Tuple

import imagehash
import numpy as np
from PIL import Image

def hash_signature(file: str, hash_size: int) -> np.ndarray:
    """
    Calculates the dhash signature of a given image using greyscale
    
    """
    pil_image = Image.open(file).convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    dhash = imagehash.dhash(pil_image, hash_size)
    signature = dhash.hash.flatten()
    pil_image.close()
    return signature

print(hash_signature(r"learning_to_hash\lsh\test_image.png", 8))