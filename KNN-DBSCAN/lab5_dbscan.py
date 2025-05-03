import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def select_sample(df, sample_size=1000):
    """Selects a sample of the specified size from the DataFrame."""
    return df.sample(n=sample_size, random_state=42)

class DBSCANClustering:
    """
    A class to perform DBSCAN clustering and evaluate its performance.

    This class encapsulates the process of loading data, applying DBSCAN clustering,
    and evaluating the resulting clusters using various metrics. It is designed to be
    flexible and configurable through a configuration dictionary.
    """

    def __init__(self, config):
        """
        Initializes the DBSCANClustering object with a configuration dictionary.

        Args:
            config (dict): A dictionary containing configuration parameters for
                           data loading, DBSCAN model, and evaluation metrics.
                           See the 'CONFIGURATION' section in the script for details.
        """
        self.config = config
        self.data = None  # Placeholder for the loaded feature data
        self.target = None # Placeholder for the target data (ground truth, optional for evaluation)
        self.dbscan_model = None # Placeholder for the trained DBSCAN model
        self.labels = None # Placeholder for the cluster labels assigned by DBSCAN
        self.feature_data = None # Placeholder to store the original feature data

    def _load_data(self):
        """
        Loads data from a CSV file specified in the configuration.

        This private method reads the CSV file, extracts feature columns and optionally
        a target column if specified in the configuration. It handles potential errors
        during file reading and column selection.

        Raises:
            FileNotFoundError: If the data file specified in 'data_path' is not found.
            KeyError: If specified feature or target columns are not found in the DataFrame.
        """
        try:
            data_path = self.config['data_config']['data_path']
            feature_columns = self.config['data_config']['feature_columns']
            target_column = self.config['data_config'].get('target_column') # Use .get() to avoid KeyError if target_column is not provided
            sample_size = self.config['data_config'].get('sample_size')

            self.data = pd.read_csv(data_path)

            if sample_size:
                self.data = select_sample(self.data, sample_size)

            # Select feature data
            try:
                self.feature_data = self.data[feature_columns].copy() # Store original feature data
            except KeyError as e:
                raise KeyError(f"Feature column(s) '{e}' not found in the data file.")

            # Select target data if specified (for evaluation purposes)
            if target_column:
                try:
                    self.target = self.data[target_column]
                except KeyError:
                    print(f"Warning: Target column '{target_column}' not found. Evaluation metrics requiring ground truth labels will be skipped.")
                    self.target = None # Set target to None if not found
            else:
                self.target = None # Explicitly set to None if no target column is configured

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found at path: {data_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def _scale_features(self):
        """
        Scales the feature data using StandardScaler.

        This private method applies standardization (scaling to zero mean and unit variance)
        to the feature data. Scaling can be important for distance-based algorithms like DBSCAN.
        Whether to scale or not is controlled by the 'scale_features' flag in the configuration.

        Returns:
            pandas.DataFrame: Scaled feature data.
        """
        if self.config['preprocessing']['scale_features']:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.feature_data)
            return pd.DataFrame(scaled_data, columns=self.feature_data.columns) # Maintain column names after scaling
        else:
            return self.feature_data # Return original data if scaling is not enabled


    def train_dbscan(self):
        """
        Trains the DBSCAN clustering model using the loaded and preprocessed data.

        This method initializes and fits the DBSCAN model using parameters specified
        in the configuration. It stores the trained model and the assigned cluster labels
        in the object's attributes.
        """
        try:
            self._load_data() # Load data before training
            processed_data = self._scale_features() # Scale features if configured

            dbscan_params = self.config['dbscan_params']
            self.dbscan_model = DBSCAN(**dbscan_params) # Initialize DBSCAN with configuration parameters
            self.labels = self.dbscan_model.fit_predict(processed_data) # Fit and predict in one step, store cluster labels

        except Exception as e:
            raise Exception(f"Error during DBSCAN training: {e}")

    def get_model_results(self):
        """
        Returns the results of the DBSCAN clustering.

        This method provides access to the cluster labels assigned by the trained DBSCAN model.

        Returns:
            numpy.ndarray: An array of cluster labels. -1 indicates noise points.
        """
        if self.labels is None:
            raise ValueError("DBSCAN model has not been trained yet. Call 'train_dbscan()' first.")
        return self.labels

    def evaluate_model_performance(self):
        """
        Evaluates the performance of the DBSCAN clustering model.

        This method calculates and prints various performance metrics for the clustering result.
        It includes both unsupervised metrics (like Silhouette Score) and supervised metrics
        (like Adjusted Rand Index) if ground truth labels are available in 'self.target'.

        The metrics to be calculated are controlled by the 'evaluation_metrics' section
        in the configuration.
        """
        if self.labels is None:
            raise ValueError("DBSCAN model has not been trained yet. Call 'train_dbscan()' first.")

        labels = self.labels
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0) # Number of clusters, ignoring noise if present
        n_noise_ = list(labels).count(-1) # Number of noise points

        print("------------------- DBSCAN Model Performance -------------------")
        print(f"Estimated number of clusters: {n_clusters_}")
        print(f"Estimated number of noise points: {n_noise_}")

        evaluation_config = self.config['evaluation_metrics']

        # Unsupervised Metrics (always calculated for DBSCAN)
        if evaluation_config.get('silhouette_score', True): # Default to True if not specified
            if n_clusters_ > 1 and n_clusters_ < len(labels): # Silhouette score is not defined if n_clusters is 1 or equals n_samples
                silhouette_avg = metrics.silhouette_score(self._scale_features(), labels)
                print(f"Silhouette Coefficient: {silhouette_avg:.3f}")
            else:
                print("Silhouette Coefficient: Not calculated (Number of clusters <= 1 or equals to number of samples)")

        if evaluation_config.get('calinski_harabasz_index', True): # Default to True if not specified
            if n_clusters_ > 1 and n_clusters_ < len(labels):
                ch_score = metrics.calinski_harabasz_score(self._scale_features(), labels)
                print(f"Calinski-Harabasz Index: {ch_score:.3f}")
            else:
                print("Calinski-Harabasz Index: Not calculated (Number of clusters <= 1 or equals to number of samples)")

        if evaluation_config.get('davies_bouldin_index', True): # Default to True if not specified
            if n_clusters_ > 1 and n_clusters_ < len(labels):
                db_score = metrics.davies_bouldin_score(self._scale_features(), labels)
                print(f"Davies-Bouldin Index: {db_score:.3f}")
            else:
                print("Davies-Bouldin Index: Not calculated (Number of clusters <= 1 or equals to number of samples)")


        # Supervised Metrics (only if target data is provided)
        if self.target is not None:
            if evaluation_config.get('adjusted_rand_index', True): # Default to True if not specified
                ari = metrics.adjusted_rand_score(self.target, labels)
                print(f"Adjusted Rand Index: {ari:.3f}")

            if evaluation_config.get('normalized_mutual_info_score', True): # Default to True if not specified
                nmi = metrics.normalized_mutual_info_score(self.target, labels)
                print(f"Normalized Mutual Information: {nmi:.3f}")
        else:
            print("Ground truth labels (target column) not provided. Supervised metrics skipped.")
        print("------------------------------------------------------------")


    def run_dbscan_clustering(self):
        """
        Orchestrates the entire DBSCAN clustering process.

        This is the main method to run the DBSCAN clustering. It trains the model,
        gets the results, and evaluates the performance based on the provided configuration.

        Returns:
            numpy.ndarray: The cluster labels assigned by DBSCAN.
        """
        try:
            self.train_dbscan()
            self.evaluate_model_performance()
            self.plot_k_distance_graph()
            cluster_labels = self.get_model_results()

            # Perform PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(self._scale_features())

            # Create scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
            plt.title('PCA Scatter Plot of DBSCAN Clustering')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Cluster')
            plt.savefig('/teamspace/studios/this_studio/output/dbscan_pca.png')
            plt.close()

            cluster_labels = self.get_model_results()

            # Create centroid table
            if isinstance(self.feature_data, pd.DataFrame): # Pass self.feature_data (original data)
                centroid_table = self.create_centroid_table(self.feature_data, cluster_labels) # Pass original feature data
                print("------------------- Centroid Table (Original Scale) -------------------")
                print(centroid_table)
                print("-----------------------------------------------------------------------")
            else:
                print("Warning: Original features is not a DataFrame. Centroid table cannot be created.")

            return cluster_labels
        except Exception as e:
            print(f"An error occurred during DBSCAN clustering process: {e}")
            return None

    def create_centroid_table(self, original_data, labels): # Expect original_data
        """
        Creates a centroid table for each cluster, showing the average feature values for each cluster
        in the *original, unscaled* feature space.

        Args:
            original_data (pandas.DataFrame): The *original, unscaled* feature data.
            labels (numpy.ndarray): The cluster labels assigned by DBSCAN.

        Returns:
            pandas.DataFrame: A DataFrame where each row represents a cluster and the
                              columns represent the average feature values of that cluster.
        """
        centroids = []
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue

            cluster_data = original_data[labels == label] # Use original_data here

            if len(cluster_data) == 0:
                print(f"Warning: No data points found for cluster {label}. Skipping centroid calculation.")
                continue

            # Calculate the mean of the cluster
            cluster_mean = cluster_data.mean()
            centroids.append(cluster_mean)

        return pd.DataFrame(centroids)


    def plot_k_distance_graph(self, k=None):
        """
        Generates and plots the k-distance graph to help estimate epsilon (eps) for DBSCAN.

        Args:
            k (int, optional): The 'k' value to use for k-nearest neighbors. If None,
                                it defaults to the 'min_samples' value from the DBSCAN configuration.
                                If 'min_samples' is not specified in the config, it defaults to 5.
        Returns:
            numpy.ndarray: Sorted distances to the k-th nearest neighbor for each point.
                           These distances are plotted in the graph.
        """
        try:
            self._load_data() # Load data if not already loaded
            processed_data = self._scale_features() # Scale features

            if k is None:
                k = self.config['dbscan_params'].get('min_samples', 5) # Default to min_samples from config or 5

            neighbors = NearestNeighbors(n_neighbors=k)
            neighbors.fit(processed_data)
            distances, indices = neighbors.kneighbors(processed_data)

            distances = np.sort(distances, axis=0)
            distances = distances[:, 1] # Get the distances to the k-th nearest neighbor

            plt.figure(figsize=(10, 6))
            plt.plot(distances)
            plt.title(f'K-distance Graph (k={k}) for Eps Estimation')
            plt.xlabel('Data Points (sorted by distance to k-th neighbor)')
            plt.ylabel(f'Distance to {k}-th Nearest Neighbor')
            plt.grid(True)
            plt.savefig('/teamspace/studios/this_studio/output/elbow.png')

            return distances # Return distances for further inspection if needed

        except Exception as e:
            print(f"Error generating k-distance graph: {e}")
            return None


# ------------------------ CONFIGURATION ------------------------
# Detailed and Flexible Configuration Dictionary:
config = {
    'data_config': {
        'data_path': '/teamspace/studios/this_studio/dataset/hb_outlier_removed.csv',  # Path to your CSV data file
        'feature_columns': [
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
        ], # List of column names to use as features
        'target_column': 'is_canceled', # Optional: Column name for ground truth cluster labels for evaluation (e.g., 'true_labels'). Set to None if not available.
        'sample_size': 10000 # Number of samples to select from the DataFrame
    },
    'preprocessing': {
        'scale_features': True, # Boolean: Whether to scale features using StandardScaler
    },
    'dbscan_params': {
        'eps': 4,             # The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        'min_samples': 5,       # The number of samples in a neighborhood for a point to be considered as a core point.
        'metric': 'euclidean',  # The metric to use when calculating distance between instances in a feature array.
                                # Options: 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'wminkowski', 'seuclidean',
                                #          'mahalanobis', 'cityblock', 'l1', 'l2', 'hamming', 'canberra', 'braycurtis',
                                #          'cosine', 'correlation', 'sqeuclidean', 'jensenshannon', 'dice', 'rogerstanimoto',
                                #          'russellrao', 'sokalmichener', 'sokalsneath', 'yule'. See sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS for more.
        'algorithm': 'auto',    # The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors.
                                # Options: 'auto', 'ball_tree', 'kd_tree', 'brute'. 'auto' will attempt to decide the most appropriate algorithm based on the values passed to fit method.
        'leaf_size': 30,        # Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree.
                                # Significant for 'ball_tree' or 'kd_tree' algorithm options.
        'p': None,              # The power of the Minkowski metric to be used to calculate distance between points. If None, then p=2 (equivalent to the Euclidean distance).
                                # For Minkowski-p metric.
        'n_jobs': -1            # The number of parallel jobs to run. -1 means using all processors.
                                # Useful for large datasets and faster computation.
    },
    'evaluation_metrics': {
        'silhouette_score': True,           # Calculate Silhouette Coefficient
        'calinski_harabasz_index': True,    # Calculate Calinski-Harabasz Index
        'davies_bouldin_index': True,       # Calculate Davies-Bouldin Index
        'adjusted_rand_index': True,        # Calculate Adjusted Rand Index (requires target_column in data_config)
        'normalized_mutual_info_score': True # Calculate Normalized Mutual Information (requires target_column in data_config)
    }
}
# --------------------- END CONFIGURATION ----------------------


if __name__ == "__main__":
    # Example Usage:
    pd.set_option('display.max_columns', None)
    try:
        clusterer = DBSCANClustering(config)
        cluster_labels = clusterer.run_dbscan_clustering()

    except Exception as e:
        print(f"Error during execution: {e}")