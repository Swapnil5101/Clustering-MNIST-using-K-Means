import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Function to compute the cosine similarity matrix between two sets of vectors
def cosine_similarity_matrix(d, mu):
    norms_d = np.linalg.norm(d, axis=1)
    norms_mu = np.linalg.norm(mu, axis=1)
    return np.dot(d, mu.T) / (np.outer(norms_d, norms_mu) + 1e-12)

# Function to initialize centroids randomly
def initialize_centroids(d, k):
    return d[np.random.choice(d.shape[0], k, replace=False)]

# Function to update centroids based on assigned data points
def update_centroids(d, assignments, k):
    centroids = np.array([np.mean(d[assignments == j], axis=0) for j in range(k)])
    return centroids

# Function representing a single iteration of the k-means algorithm
def kmeans_iteration(d, mu):
    r = cosine_similarity_matrix(d, mu)
    assignments = np.argmax(r, axis=1)
    return assignments

# Function to check if k-means algorithm has converged
def kmeans_converged(mu, old_mu, tol=1e-6):
    return np.all(np.abs(mu - old_mu) < tol)

# Function to perform the k-means clustering algorithm
def kmeans(d, k, max_itr=500):
    mu = initialize_centroids(d, k)
    old_mu = np.copy(mu)
    wcss_values = []  # within-cluster sum of squares
    for t in range(max_itr):
        assignments = kmeans_iteration(d, mu)
        mu = update_centroids(d, assignments, k)
        wcss_values.append(wcss(d, mu, assignments))
        if kmeans_converged(mu, old_mu):
            return mu, assignments, wcss_values
        old_mu = np.copy(mu)
    return mu, assignments, wcss_values

# Function to calculate the within-cluster sum of squares
def wcss(d, mu, assignments):
    wcss = np.sum([np.sum(np.linalg.norm(d[assignments == i] - mu[i], axis=1) ** 2) for i in range(len(mu))])
    return wcss

# Function to plot clusters of images
def plot_clusters(images, assignments, k):
    grouped_images = {i: [] for i in range(k)}
    for i in range(len(assignments)):
        k = assignments[i]
        grouped_images[k].append(images[i])
    for k, image_list in grouped_images.items():
        plt.figure(figsize=(10, 5))
        for i, image in enumerate(image_list[:10]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(image.reshape(28, 28), cmap='gray')
            plt.axis('off')
        plt.suptitle(f'Cluster {k}')
        plt.show()

# Main function
def main():
    mnist = pd.read_csv("/content/sample_data/mnist_train_small.csv")
    k_list = [4, 7, 10, 13]
    wcss_list = []
    images = mnist.drop(mnist.columns[0], axis=1).to_numpy() / 255.0
    for k in k_list:
        print(f"At k = {k}")
        final_centroids, assignments, wcss_values = kmeans(images, k)
        wcss_list.append(wcss_values[-1])
        plot_clusters(images, assignments, k)
    plt.plot(k_list, wcss_list, marker='o')
    plt.title('wss vs k')
    plt.xlabel('k')
    plt.ylabel('WSS')
    plt.show()

if __name__ == "__main__":
    main()
