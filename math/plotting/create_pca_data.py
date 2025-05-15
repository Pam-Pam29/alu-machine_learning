import numpy as np
from sklearn import datasets
import os

# Load the iris dataset
iris = datasets.load_iris()
data = iris.data  # Shape: (150, 4) - Features of iris flowers
labels = iris.target  # Shape: (150,) - Species labels (0, 1, 2)

# Save the data in the required format
np.savez("pca.npz", data=data, labels=labels)

print("Created pca.npz file with Iris dataset")
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels unique values: {np.unique(labels)}")