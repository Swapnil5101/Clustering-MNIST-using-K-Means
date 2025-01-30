import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture as GM
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load MNIST data and normalize."""
    mnist = pd.read_csv(file_path)
    images = mnist.drop(mnist.columns[0], axis=1).to_numpy()

    return images / 255.0

def perform_gmm_clustering(images, num_components, num_clusters):
    """Perform GMM clustering and visualize results."""
    U, S, VT = np.linalg.svd(images, full_matrices=False)
    principle_component = images @ VT.T[:, :num_components]

    print(f"No of Principle Components: {num_components}")
    print(f"No of Clusters: {num_clusters}")

    gmm = GM(n_components=num_clusters)
    gmm.fit(principle_component)
    output = gmm.predict(principle_component)

    # Visualize clusters
    visualize_clusters(images, output, num_clusters)

    # Scatter plot
    plt.scatter(principle_component[:, 0], principle_component[:, 1], c=output, cmap='viridis')
    plt.title('GMM Clustering')
    plt.show()

def visualize_clusters(images, output, num_clusters):
    """Visualize clustered images."""
    plt.figure(figsize=(12, 15))
    for label in range(num_clusters):
        cluster_images = images[output == label]
        for k in range(5):
            plt.subplot(num_clusters, 5, label * 5 + k + 1)
            plt.imshow(cluster_images[k].reshape(28, 28), cmap=plt.cm.gray)
            plt.title(f"Cluster {label}")
            plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    file_path = "/content/sample_data/mnist_train_small.csv"
    images = load_data(file_path)

    num_components_list = [32, 64, 128]
    num_clusters_list = [4, 7, 10]

    for num_components in num_components_list:
        for num_clusters in num_clusters_list:
            perform_gmm_clustering(images, num_components, num_clusters)

if __name__ == "__main__":
    main()
