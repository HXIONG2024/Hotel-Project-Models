import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import random

# Set seaborn style
sns.set(style="whitegrid")

class HotelBookingClustering:
    """
    Performs hotel booking data clustering using K-Means on hotel booking data (NO SCALING).
    Includes Elbow method for optimal k, NNCI for clustering tendency assessment,
    PCA for visualization, and plotting.
    """

    def __init__(self, data_path, feature_names):
        """
        Initializes the HotelBookingClustering object.

        Args:
            data_path (str): Path to the CSV file containing hotel booking data.
            feature_names (list): List of feature names to be used for clustering.
        """
        self.data_path = data_path
        self.feature_names = feature_names
        self.df = self._load_data()
        self.X_original = self.df[self.feature_names].values
        self.X = self.df[self.feature_names]

        # --- SCALING COMPLETELY REMOVED - Clustering will be performed on original data ---
        self.X_scaled = self.X_original.copy()  # X_scaled is now just a copy of original data


    def _load_data(self):
        """Loads the data from the specified CSV file."""
        try:
            df = pd.read_csv(self.data_path)
            print(f"Original data shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: CSV file not found at {self.data_path}")
            raise

    # --- _preprocess_data() method REMOVED as no scaling is performed ---


    def assess_clustering_tendency_nnci(self, k_neighbors=5, n_random_points=None, random_state=42):
        """
        Assess the clustering tendency of the data using Nearest Neighbor Clustering Index (NNCI).

        NNCI compares the average k-th nearest neighbor distance in the data to that of a
        randomly distributed dataset. Lower NNCI values suggest stronger clustering.

        Args:
            k_neighbors (int): The number of nearest neighbors to consider (k).
            n_random_points (int, optional): Number of random points to generate. If None, defaults to the number of real data points.
            random_state (int): Random seed for reproducibility.

        Returns:
            float: Nearest Neighbor Clustering Index (NNCI) value.
                   - Lower values (typically below 1, and can be negative depending on normalization): Indicate clustering tendency.
                   - Values around 1: Suggest data is similar to random distribution (no strong clustering).
                   - Higher values (above 1): May indicate regular or grid-like data (uncommon in practice for clustering).
        """
        np.random.seed(random_state)
        X_scaled = self.X_scaled # Now this is just X_original (no scaling)
        X_original = self.X_original # Still need original for range calculation
        n, d = X_scaled.shape

        if n_random_points is None:
            n_random = n
        else:
            n_random = n_random_points


        # 1. Calculate average k-NN distance for real data (using ORIGINAL data - NO SCALING)
        nbrs_real = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(X_scaled)
        distances_real, _ = nbrs_real.kneighbors(X_scaled)
        avg_real_dist = np.mean(distances_real[:, k_neighbors])


        # 2. Generate random data in the ORIGINAL feature space (NO SCALING needed for random data either)
        min_vals_original = np.min(X_original, axis=0)
        max_vals_original = np.max(X_original, axis=0)
        X_random_original = np.random.uniform(low=min_vals_original, high=max_vals_original, size=(n_random, d))
        X_scaled_random = X_random_original.copy() # No scaling for random data either


        # 3. Calculate average k-NN distance for random data (using ORIGINAL scale random data)
        nbrs_random = NearestNeighbors(n_neighbors=k_neighbors).fit(X_scaled_random)
        distances_random, _ = nbrs_random.kneighbors(X_scaled_random)
        avg_random_dist = np.mean(distances_random[:, k_neighbors-1])

        # 4. Calculate NNCI
        if avg_random_dist == 0:
            NNCI = 0.0
        else:
            NNCI = avg_real_dist / avg_random_dist


        print(f"Nearest Neighbor Clustering Index (NNCI): {NNCI:.4f} (NO SCALING)")
        if NNCI < 0.8:
            print("Indicates a potential clustering tendency.")
        elif NNCI < 1.2:
            print("Indicates weak or uncertain clustering tendency (data may be somewhat random-like).")
        else:
            print("Indicates no strong clustering tendency.")

        return NNCI


    def elbow_method(self, max_k=10, random_state=42, save_path='/teamspace/studios/this_studio/output/elbow_method.png'):
        """
        Performs the Elbow Method to find the optimal number of clusters (k).

        Args:
            max_k (int): The maximum number of clusters to test.
            random_state (int): Random seed for reproducibility.
            save_path (str): Path to save the Elbow method plot.

        Returns:
            list: Inertia values for each k.
        """
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=random_state)
            kmeans.fit(self.X_scaled) # Fit KMeans on unscaled data
            inertias.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.title('Elbow Method for Optimal k (NO SCALING)')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.xticks(range(1, max_k + 1))
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        print(f"Elbow method plot saved to: {save_path}")
        return inertias


    def perform_clustering_and_visualization(self, k, save_plots=True, output_path='/teamspace/studios/this_studio/output/'):
        """
        Performs K-Means clustering, PCA for visualization, and generates cluster plots.

        Args:
            k (int): The number of clusters for K-Means.
            save_plots (bool): Whether to save the cluster plots to files.
            output_path (str): Path to save the output plots.
        """
        # K-Means Clustering (on UNSCALED data)
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
        clusters = kmeans.fit_predict(self.X_scaled) # Fit and predict on UNSCALED data
        centroids = kmeans.cluster_centers_

        # Centroids are already in original scale (no inverse transform needed)
        centroids_original = centroids.copy()

        # PCA for Visualization (on UNSCALED data)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled) # Fit PCA on UNSCALED data
        centroids_pca = pca.transform(centroids)

        self._plot_final_clusters(X_pca, clusters, centroids_pca, centroids_original, k, save=save_plots, output_path=output_path)


    def _plot_final_clusters(self, X_pca, clusters, centroids_pca, centroids_original, k, save=False, output_path='/teamspace/studios/this_studio/output/'):
        """
        Generates and saves the final cluster plot.

        Args:
            X_pca (numpy.ndarray): PCA-transformed data.
            clusters (numpy.ndarray): Cluster labels.
            centroids_pca (numpy.ndarray): PCA-transformed centroids.
            centroids_original (numpy.ndarray): Centroids in original feature space.
            k (int): Number of clusters.
            save (bool): Whether to save the plot.
            output_path (str): Path to save the plot file.
        """
        plt.figure(figsize=(8, 6))
        plt.title(f'Final K-Means Clustering (PCA Projection) - k={k} (NO SCALING)')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        if k > len(colors):
            raise ValueError(f"Not enough colors defined for k={k}. Increase the 'colors' list.")

        for i in range(k):
            points = X_pca[clusters == i]
            plt.scatter(points[:, 0], points[:, 1], s=30, c=colors[i], label=f'Cluster {i+1}')

        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=200, c='yellow', marker='X', edgecolor='k', label='Centroids')
        plt.legend()

        if save:
            filename = f'{output_path}final_clusters_k_{k}_NO_SCALING.png'
            plt.savefig(filename)
            print(f"Final clusters plot saved as {filename}")
            plt.close()
        else:
            plt.show()

        print(f"Number of clusters (k): {k}")
        print(f"Centroids (in PCA space):\n{centroids_pca}")
        print("\nCentroids (in original feature space):")
        pd.set_option('display.max_columns', None)
        centroids_df = pd.DataFrame(centroids_original, columns=self.feature_names)
        centroids_df = centroids_df.sort_values(by='lead_time')
        print(centroids_df)



# --- Configuration and Usage ---
if __name__ == "__main__":
    # 1. Define data path and feature names
    DATA_PATH = "/teamspace/studios/this_studio/dataset/hb_outlier_removed.csv"
    FEATURE_NAMES = [
        'lead_time',
        'previous_cancellations',
        'is_canceled',
        'total_of_special_requests',
        'stays_in_week_nights',
        'stays_in_weekend_nights',
        'is_repeated_guest',
        'booking_changes',
        'required_car_parking_spaces',
        'is_city_hotel',
        'market_segment_Direct',
        'reserved_room_type_A',
        'adr'
    ]
    OUTPUT_DIR = '/teamspace/studios/this_studio/output/'

    # 2. Instantiate the HotelBookingClustering class
    clustering_processor = HotelBookingClustering(data_path=DATA_PATH, feature_names=FEATURE_NAMES)

    # --- Diagnostic Checks (Essential Checks Kept) ---
    print("\n--- Diagnostic Checks (Essential) ---")

    # 1. Feature Variances in Original Data
    feature_variances = np.var(clustering_processor.X_scaled, axis=0)
    print("\nFeature Variances in Original Data (NO SCALING):")
    for i, feature_name in enumerate(FEATURE_NAMES):
        print(f"Feature '{feature_name}': {feature_variances[i]:.4f}")

    # 2. Duplicate Row Count in Original Data (Selected Features)
    df_scaled = pd.DataFrame(clustering_processor.X_scaled, columns=FEATURE_NAMES)
    duplicate_count_scaled = df_scaled.duplicated().sum()
    print(f"\nNumber of Duplicate Rows in Original Data (NO SCALING - Selected Features): {duplicate_count_scaled}")

    # 3. Unique Value Counts for Each Feature in ORIGINAL DataFrame
    print("\nUnique Value Counts for Each Feature (Original Data):")
    for feature_name in FEATURE_NAMES:
        unique_count = clustering_processor.df[feature_name].nunique()
        print(f"Feature '{feature_name}': {unique_count} unique values")


    # --- 3. Assess Clustering Tendency using NNCI ---
    print("\n--- Assessing Clustering Tendency using NNCI (NO SCALING) ---")
    nnci_stat = clustering_processor.assess_clustering_tendency_nnci(k_neighbors=5)
    print(f"NNCI Statistic Value: {nnci_stat:.4f}")


    # 4. Run the Elbow Method to determine optimal k
    print("\nRunning Elbow Method...")
    inertia_values = clustering_processor.elbow_method(max_k=10, save_path=f'{OUTPUT_DIR}elbow_method.png')
    print(f"Inertia values for each k: {inertia_values}")
    print("Elbow Method completed and plot saved.")

    # 5. Choose the optimal k based on the Elbow plot inspection
    #    (Inspect the 'elbow_method.png' in the output directory)
    chosen_k = 5  # Replace with the optimal k you determine from the Elbow plot

    # 6. Perform clustering and visualization with the chosen k
    print(f"\nPerforming K-Means clustering and visualization with k={chosen_k}...")
    clustering_processor.perform_clustering_and_visualization(k=chosen_k, save_plots=True, output_path=OUTPUT_DIR)
    print(f"Clustering and visualization with k={chosen_k} completed and plots saved.")

    # 7. Example to test a range of k values (optional)
    # print("\nTesting clustering for a range of k values...")
    # for k_val in range(2, 7):  # Example: Test k values from 2 to 6
    #     print(f"\n--- Clustering with k={k_val} ---")
    #     clustering_processor.perform_clustering_and_visualization(k=k_val, save_plots=True, output_path=OUTPUT_DIR)
    # print("Range testing completed.")