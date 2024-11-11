import numpy as np
from sklearn.preprocessing import normalize
from PIL import Image
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


def load_image(image):
  image = Image.fromarray(image).convert("L")  # Convert to grayscale
  image_array = np.array(image)
  return image_array.flatten()  # Flatten the image array

def spectral_hashing(data, n_bits):
    """
    Applies Spectral Hashing to the input data.

    Args:
        data: The input data as a NumPy array.
        n_bits: The number of bits to use for hashing.

    Returns:
        A NumPy array containing the binary hash codes for each data point.
    """

    # 1. Center the data
    data = data - np.mean(data, axis=0)

    # 2. Compute the PCA transformation
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 3. Select the top eigenvectors
    top_eigenvectors = eigenvectors[:, -n_bits:]

    # 4. Project the data onto the eigenvectors
    projected_data = data.dot(top_eigenvectors)

    # 5. Generate binary hash codes using the sign function
    binary_codes = np.sign(projected_data)
    binary_codes[binary_codes == -1] = 0  # Replace -1 with 0
    binary_codes = binary_codes.astype(int)  # Convert to integers

    return binary_codes


def hamming_distance(hash1, hash2):
    return np.sum(np.abs(hash1 - hash2))


def query_images(query_hash, hash_codes_db, top_k=5):
  distances = {}
  for image_path, hash_code in hash_codes_db.items():
    distances[image_path] = hamming_distance(query_hash, hash_code)

  sorted_distances = sorted(distances.items(), key=lambda item: item[1])
  print(sorted_distances)
  # Return both image path and distance
  return [(image_path, distance) for image_path, distance in sorted_distances[:top_k]]


# Sample data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

new_train_images = train_images[:60000]
data = np.array([load_image(img) for img in new_train_images])

# Generate hash codes with 16 bits
hash_codes = spectral_hashing(data, n_bits=32)

hash_codes_table = {}
for i, hash_code in enumerate(hash_codes):
    hash_codes_table[i] = hash_code

# Generate hash code for query image
query_hash = hash_codes[0]
 
 # Find similar images
similar_images = query_images(query_hash, hash_codes_table)
 
# Display similar images using Matplotlib
fig, axes = plt.subplots(1, len(similar_images) + 1, figsize=(15, 5))

# Display the query image first
axes[0].imshow(new_train_images[0].reshape(32, 32, 3))  
axes[0].set_title("Query Image")
axes[0].axis('off')

# Display similar images in the remaining subplots
for i, (image_index, distance) in enumerate(similar_images):
  image = new_train_images[image_index].reshape(32, 32, 3)
  axes[i + 1].imshow(image)  # Shift index by 1
  # Include distance in the title
  axes[i + 1].set_title(f"Similar Image {i + 1}\n(Distance: {distance})")  
  axes[i + 1].axis('off')
    
plt.show()