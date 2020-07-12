import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()

model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8, 3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')

for i in range(10):
    # Initialize subplots in a grid of 2X5, at i+1th position
    ax = fig.add_subplot(2, 5, 1 + i)

    # Display images
    ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

new_samples = np.array([
    [0.30, 3.05, 6.03, 6.86, 5.95, 1.07, 0.00, 0.00, 6.02, 7.62, 6.25, 4.20, 7.09, 5.87, 0.00, 0.00, 5.87, 2.67, 0.23,
     0.00, 3.58, 7.62, 1.60, 0.00, 0.00, 0.00, 0.00, 0.00, 2.06, 7.62, 2.29, 0.00, 0.00, 0.00, 0.00, 0.61, 6.56, 6.56,
     0.69, 0.00, 0.00, 0.00, 1.14, 6.25, 7.55, 2.44, 0.76, 0.31, 0.00, 0.00, 6.02, 7.62, 7.62, 7.62, 7.62, 7.24, 0.00,
     0.00, 1.38, 2.29, 2.29, 2.29, 2.52, 3.81],
    [0.00, 0.00, 1.98, 7.09, 7.63, 5.80, 0.15, 0.00, 0.00, 0.76, 6.56, 6.94, 3.28, 7.62, 3.73, 0.00, 0.00, 5.19, 7.24,
     1.98, 0.00, 5.87, 5.95, 0.00, 0.00, 6.86, 4.42, 0.00, 0.00, 4.27, 6.86, 0.00, 0.00, 6.86, 3.81, 0.00, 0.00, 4.35,
     6.86, 0.00, 0.00, 6.86, 5.34, 1.52, 0.76, 6.10, 5.87, 0.00, 0.00, 4.19, 7.62, 7.62, 7.62, 7.62, 2.98, 0.00, 0.00,
     0.00, 1.30, 2.06, 2.29, 2.06, 0.00, 0.00],
    [0.00, 2.75, 5.34, 5.34, 3.51, 0.00, 0.00, 0.00, 0.00, 2.75, 5.34, 6.48, 7.62, 1.30, 0.00, 0.00, 0.00, 0.00, 0.00,
     1.60, 7.62, 1.52, 0.00, 0.00, 0.00, 0.00, 0.00, 1.60, 7.62, 1.52, 0.00, 0.00, 0.00, 0.00, 1.30, 5.87, 7.55, 0.99,
     0.00, 0.00, 0.00, 5.11, 7.62, 7.24, 3.36, 0.00, 0.00, 0.00, 1.37, 7.62, 7.62, 7.62, 7.62, 7.62, 7.62, 5.65, 0.23,
     2.98, 3.05, 3.05, 3.05, 3.05, 3.05, 1.98],
    [0.00, 1.68, 5.34, 5.34, 5.34, 2.75, 0.00, 0.00, 0.00, 1.68, 5.34, 5.34, 6.56, 7.63, 1.07, 0.00, 0.00, 0.00, 0.00,
     0.77, 4.88, 7.62, 2.29, 0.00, 0.00, 0.00, 5.72, 7.47, 7.62, 6.79, 0.61, 0.00, 0.00, 0.00, 4.35, 5.34, 6.25, 7.62,
     3.59, 0.00, 0.00, 0.00, 0.00, 0.00, 3.89, 7.62, 3.05, 0.00, 0.00, 3.82, 6.10, 6.86, 7.62, 5.95, 0.00, 0.00, 0.00,
     2.75, 4.57, 4.50, 3.43, 1.37, 0.00, 0.00]
])

new_labels = model.predict(new_samples)

for i in range(len(new_labels)):
    if new_labels[i] == 0:
        print(0, end='')
    elif new_labels[i] == 1:
        print(9, end='')
    elif new_labels[i] == 2:
        print(2, end='')
    elif new_labels[i] == 3:
        print(1, end='')
    elif new_labels[i] == 4:
        print(6, end='')
    elif new_labels[i] == 5:
        print(8, end='')
    elif new_labels[i] == 6:
        print(4, end='')
    elif new_labels[i] == 7:
        print(5, end='')
    elif new_labels[i] == 8:
        print(7, end='')
    elif new_labels[i] == 9:
        print(3, end='')
