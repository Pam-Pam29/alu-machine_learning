import numpy as np
from sklearn import datasets
import os

iris = datasets.load_iris()
data = iris.data  
labels = iris.target  

np.savez("pca.npz", data=data, labels=labels)

print("Created pca.npz file with Iris dataset")
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Labels unique values: {np.unique(labels)}")