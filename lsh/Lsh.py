import sys
import os
import os.path

import hashlib as hl
import cv2 as cv
import numpy as np

import matplotlib.pyplot as plt

from google.colab.patches import cv2_imshow

from scipy.spatial import distance

from collections import defaultdict
import imagehash
from datasketch import MinHash, MinHashLSH
from PIL import Image

# CIfar10
from tensorflow.keras.datasets import cifar10

def compute_phash(image, size: tuple):
  img = Image.fromarray(image)
  phash = imagehash.phash(img)
  return str(phash)

def compute_min_hash(images):
  signatures = []
  for image in images:
    m = MinHash(num_perm=65)
    phash = compute_phash(image, (32, 32))

    for bit in phash:
      m.update(bit.encode('utf8'))
    signatures.append(m)
  return signatures


(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

new_train_images = train_images[:10000]

signatures = compute_min_hash(new_train_images)

lsh = MinHashLSH(threshold=0.88, num_perm=65)
for i, signature in enumerate(signatures):
    lsh.insert(i, signature)

query_result = lsh.query(signatures[0])

num_similar_images = len(query_result) + 1

fig, axes = plt.subplots(1, num_similar_images, figsize=(20, 5))

axes[0].imshow(new_train_images[0])
axes[0].set_title('Queried Image')
axes[0].axis('off')

for i, result in enumerate(query_result):
    axes[i + 1].imshow(new_train_images[result])
    axes[i + 1].set_title(f'Similar Image {i + 1}')
    axes[i + 1].axis('off')

plt.tight_layout()
plt.show()