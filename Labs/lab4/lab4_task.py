import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from scipy.stats import mode


def let_me_see(X, k, labels, centroids, is_class=False):
    if is_class:
        title = 'Class'
    else:
        title = 'Cluster'
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    markers = ['o', '^', 's', 'v', 'p', '*']

    # two subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # first subplot: plot along 1st and 2nd features
    plt.subplot(1, 2, 1)
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], marker=markers[i], label=f'{title} {i + 1}')

    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='k', marker='*', label='Centroids')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()

    # second subplot: plot along 3rd and 4th features
    plt.subplot(1, 2, 2)
    for i in range(k):
        plt.scatter(X[labels == i, 2], X[labels == i, 3], c=colors[i], marker=markers[i], label=f'{title} {i + 1}')

    if centroids is not None:
        plt.scatter(centroids[:, 2], centroids[:, 3], c='k', marker='*', label='Centroids')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend()

    # show the plot
    plt.show()


def align_labels(y_true, y_pred):
    aligned_y_pred = np.zeros_like(y_pred)
    for i in range(3):
        mask = (y_pred == i)
        aligned_y_pred[mask] = mode(y_true[mask])[0]
    return aligned_y_pred


class KMeans:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.first_iter = True

    def calculate_distance(self, X, centroids):
        # TODO 1.1
        # Calculate the distance between each data point and each centroid
        # Hint: You may consider using
        # np.newaxis: https://numpy.org/doc/stable/reference/arrays.indexing.html
        # np.sum: https://numpy.org/doc/stable/reference/generated/numpy.sum.html
        # np.sqrt: https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
        # in case you don't know how to use them, here is an example:
        # consider print the shape of X and centroids
        # print(X.shape)
        # print(centroids.shape)
        distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))

        if self.first_iter:
            print(f"distances[0:5]:\n{distances[0:5]}")  # print the first 5 distances
            # expected results:
            # distances[0:5]:
            # [[4.63680925 0.80622577 6.21128006]
            # [4.70212718 1.34164079 6.26178888]
            # [4.86826458 1.28452326 6.44670458]
            # [4.76025209 1.42478068 6.32297398]
            # [4.68614981 0.78102497 6.26578008]]

        return distances

    def assign_labels(self, distances):
        # TODO 1.2
        # Assign each data point to the nearest centroid
        # Hint: You may consider using
        # np.argmin: https://numpy.org/doc/stable/reference/generated/numpy.argmin.html
        labels = np.argmin(distances, axis=1)

        if self.first_iter:
            print(f"labels[0:5]:\n{labels[0:5]}")
            # print the first 5 labels
            # expected results:
            # labels[0:5]:
            # [1 1 1 1 1]

        return labels

    def update_centroids(self, X, centroids, labels):
        # TODO 1.3
        # Update the centroids based on the mean of the data points assigned to them
        # Hint: You may consider using
        # np.mean: https://numpy.org/doc/stable/reference/generated/numpy.mean.html
        np.array([np.mean(X[labels == k], axis=0) for k in range(self.n_clusters)])

        if self.first_iter:
            print(f"centroids:\n{centroids}")  # print the centroids for the first iteration
            # expected results:
            # centroids:
            # [[6.17045455 2.86477273 4.80909091 1.66022727]
            # [5.00566038 3.36981132 1.56037736 0.29056604]
            # [7.57777778 3.1 6.42222222 2.04444444]]
        return centroids

    def fit(self, X):
        # Step 1: Initialize K centroids randomly
        centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]
        for i in range(self.max_iter):
            # Step 2: Calculate the distance
            distances = self.calculate_distance(X, centroids)
            # Step 3: Assign each data point to the nearest centroid
            labels = self.assign_labels(distances)
            # Step 4: Update the centroids based on the mean of the data points assigned to them
            centroids = self.update_centroids(X, centroids, labels)
            self.first_iter = False
        print(f"final centroids:\n{centroids}")  # print the final centroids
        # expected results:
        # final centroids:
        # Final centroids:
        # [[5.9016129 2.7483871 4.39354839 1.43387097]
        # [5.006 3.428 1.462 0.246 ]
        # [6.85 3.07368421 5.74210526 2.07105263]]
        print(f'final labels:\n{labels}')  # print the final labels
        # expected results:
        # final labels:
        # [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 0 2 2 2 2 0 2 2 2 2 2 2 0 0 2 2 2 2 0 2 0 2 0 2 2 0 0 2 2 2 2 2 0 2 2 2 2 0 2 2 2 0 2 2 2 0 2 2 0]

        return labels, centroids


############ Implement sse evalution function for clustering result ############
# sum of squared errors (SSE)
def sse(X, labels, centroids):
    # TODO 2.1
    # Hint: You may consider using
    # np.sum(...): https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    sse = np.sum(np.sum((X - centroids[labels]) ** 2))
    return sse


############ Implement acc evalution function for classification result ############
# accuracy (ACC)
def accuracy(y, y_pred):
    # TODO 2.2
    # Hint: You may consider using
    # np.sum(...): https://numpy.org/doc/stable/reference/generated/numpy.sum.html
    acc = np.mean(y == y_pred)
    return acc


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    # 4 features: Petal Length, Petal Width, Sepal Length, Sepal width
    y = iris.target
    # 0, 1, 2, three Classes(Species)
    np.random.seed(23333)
    ############ show the dataset ############
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    print(f'X[0:5]:\n{X[0:5]}')
    print(f'y[0:5]:\n{y[0:5]}')
    k = 3
    my_kmeans = KMeans(n_clusters=k)
    labels, centroids = my_kmeans.fit(X)
    let_me_see(X, k, labels, centroids)
