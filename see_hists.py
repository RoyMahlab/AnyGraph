import numpy as np
import matplotlib.pyplot as plt

# Example matrices (replace with your actual matrices)
matrix1 = np.random.normal(0, 1, (50, 50))
matrix2 = np.random.normal(1, 0.5, (40, 60))
matrix3 = np.random.normal(-0.5, 1.5, (30, 70))

# Flatten matrices to get 1D arrays of values
data1 = matrix1.flatten()
data2 = matrix2.flatten()
data3 = matrix3.flatten()

# Plotting overlaid histograms
bins=100
plt.figure(figsize=(10, 6))
plt.hist(data1, bins=bins, alpha=0.5, label='Matrix 1', color='blue', edgecolor='black')
plt.hist(data2, bins=bins, alpha=0.5, label='Matrix 2', color='green', edgecolor='black')
plt.hist(data3, bins=bins, alpha=0.5, label='Matrix 3', color='red', edgecolor='black')

# Adding titles and labels
plt.title('Overlaid Histogram of Matrices')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.savefig("histograms.png")
