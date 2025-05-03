"""
Hierarchical Clustering Analysis Tool

This script provides a general-purpose hierarchical clustering implementation
that can be applied to any dataset. It includes functionality for data preprocessing,
cluster analysis, visualization, and evaluation.

Author: Harry
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
import os
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

class HierarchicalClusteringAnalyzer:
    """
    A class for performing hierarchical clustering analysis on datasets.
    
    This class provides methods for data preprocessing, clustering, visualization,
    and evaluation of hierarchical clustering results.
    """
    
    def __init__(self, config):
        """
        Initialize the HierarchicalClusteringAnalyzer with configuration parameters.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary containing parameters for the analysis:
            - feature_columns: List of column names to use as features
            - n_clusters: Number of clusters to form
            - linkage_method: Method for calculating linkage ('ward', 'complete', 'average', 'single')
            - affinity: Metric used to compute linkage ('euclidean', 'manhattan', 'cosine', etc.)
            - plot_features: List of two feature names to use for 2D visualization
            - output_dir: Directory to save output files (optional)
        """
        self.config = config
        self.df = None
        self.X_scaled = None
        self.clusters = None
        self.linkage_matrix = None
        self.silhouette_avg = None
        self.pca = None
        self.X_pca = None
        
        # Create output directory if specified and doesn't exist
        if 'output_dir' in self.config and not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
    
    def load_data(self, data_source, **kwargs):
        """
        Load data from various sources.
        
        Parameters:
        -----------
        data_source : str or pandas.DataFrame
            If str: Path to the data file (CSV, Excel, etc.)
            If DataFrame: Already loaded DataFrame
        **kwargs : dict
            Additional arguments to pass to pandas read functions
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        if isinstance(data_source, pd.DataFrame):
            self.df = data_source.copy()
        elif isinstance(data_source, str):
            if data_source.endswith('.csv'):
                self.df = pd.read_csv(data_source, **kwargs)
            elif data_source.endswith(('.xls', '.xlsx')):
                self.df = pd.read_excel(data_source, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {data_source}")
        else:
            raise TypeError("data_source must be a DataFrame or a file path string")
        
        print(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns")
        return self
    
    def preprocess_data(self):
        """
        Preprocess the data by selecting features and standardizing them.
        
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        # Print debugging information
        print("\nDEBUGGING INFO:")
        print(f"Total columns in dataset: {len(self.df.columns)}")
        print(f"First 5 columns: {list(self.df.columns[:5])}")
        print(f"Features selected for clustering: {self.config['feature_columns']}")
        
        # Verify that all feature columns exist in the dataframe
        missing_columns = [col for col in self.config['feature_columns'] if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"The following feature columns are not in the dataset: {missing_columns}. "
                             f"Available columns are: {list(self.df.columns)}")
        
        # Select features for clustering
        features = self.df[self.config['feature_columns']]
        print(f"Shape of selected features: {features.shape}")
        
        # Handle missing values if any
        if features.isnull().any().any():
            print("Warning: Missing values detected. Filling with column means.")
            features = features.fillna(features.mean())
        
        # Standardize the features
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(features)
        print(f"Shape of scaled features: {self.X_scaled.shape}")
        
        print(f"Data preprocessed: {len(self.config['feature_columns'])} features standardized")
        return self
    
    def generate_linkage_matrix(self):
        """
        Generate the linkage matrix for hierarchical clustering.
        
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        self.linkage_matrix = linkage(
            self.X_scaled, 
            method=self.config['linkage_method']
        )
        print(f"Linkage matrix generated using {self.config['linkage_method']} method")
        return self
    
    def plot_dendrogram(self, figsize=(12, 8), save_fig=False):
        """
        Plot the dendrogram to visualize hierarchical clustering.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size as (width, height) in inches
        save_fig : bool, optional
            Whether to save the figure to disk
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        plt.figure(figsize=figsize)
        dendrogram(
            self.linkage_matrix,
            orientation='top',
            distance_sort='descending',
            show_leaf_counts=True
        )
        plt.title('Dendrogram - Hierarchical Clustering')
        plt.xlabel('Sample Index')
        plt.ylabel('Euclidean Distance')
        
        if save_fig and 'output_dir' in self.config:
            plt.savefig(os.path.join(self.config['output_dir'], 'dendrogram.png'), dpi=300)
        
        plt.show()
        return self
    
    def perform_clustering(self):
        """
        Perform hierarchical clustering on the preprocessed data.
        
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        # Initialize Hierarchical Agglomerative Clustering
        # The correct parameter combination depends on the linkage method
        if self.config['linkage_method'] == 'ward':
            # Ward linkage only works with euclidean distance
            hc = AgglomerativeClustering(
                n_clusters=self.config['n_clusters'],
                linkage=self.config['linkage_method']
            )
        else:
            # For other linkage methods, we can specify the affinity
            hc = AgglomerativeClustering(
                n_clusters=self.config['n_clusters'],
                affinity=self.config['affinity'],
                linkage=self.config['linkage_method']
            )
        
        # Fit and predict cluster assignments
        self.clusters = hc.fit_predict(self.X_scaled)
        
        # Add cluster assignments to the original dataframe
        self.df['Cluster'] = self.clusters
        
        print(f"Clustering completed with {self.config['n_clusters']} clusters")
        return self
    
    def evaluate_clustering(self):
        """
        Evaluate clustering performance using Silhouette Score.
        
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        self.silhouette_avg = silhouette_score(self.X_scaled, self.clusters)
        print(f"Silhouette Score: {self.silhouette_avg:.4f}")
        return self
    
    def apply_pca(self, n_components=2):
        """
        Apply PCA to the scaled data for dimensionality reduction and visualization.
        
        Parameters:
        -----------
        n_components : int, optional
            Number of principal components to keep
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        
        explained_variance = self.pca.explained_variance_ratio_
        print(f"PCA applied with {n_components} components")
        print(f"Explained variance ratio: {explained_variance}")
        print(f"Total variance explained: {sum(explained_variance):.4f}")
        
        return self
    
    def visualize_clusters_pca(self, figsize=(10, 8), save_fig=False):
        """
        Visualize the clusters using the first two principal components.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size as (width, height) in inches
        save_fig : bool, optional
            Whether to save the figure to disk
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        if self.X_pca is None or self.X_pca.shape[1] < 2:
            raise ValueError("PCA with at least 2 components must be applied first")
        
        # Create a temporary DataFrame with PCA results and cluster labels
        pca_df = pd.DataFrame({
            'PC1': self.X_pca[:, 0],
            'PC2': self.X_pca[:, 1],
            'Cluster': self.clusters
        })
        
        plt.figure(figsize=figsize)
        sns.scatterplot(
            x='PC1',
            y='PC2',
            hue='Cluster',
            palette='Set1',
            data=pca_df,
            s=100
        )
        plt.title(f'Hierarchical Clustering with PCA (k={self.config["n_clusters"]})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(title='Cluster')
        
        if save_fig and 'output_dir' in self.config:
            plt.savefig(os.path.join(self.config['output_dir'], 'clusters_pca.png'), dpi=300)
        
        plt.show()
        return self
    
    def visualize_clusters(self, figsize=(10, 8), save_fig=False):
        """
        Visualize the clusters in a 2D scatter plot.
        
        Parameters:
        -----------
        figsize : tuple, optional
            Figure size as (width, height) in inches
        save_fig : bool, optional
            Whether to save the figure to disk
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        if len(self.config['plot_features']) != 2:
            raise ValueError("Exactly 2 features must be specified for visualization")
        
        plt.figure(figsize=figsize)
        sns.scatterplot(
            x=self.config['plot_features'][0],
            y=self.config['plot_features'][1],
            hue='Cluster',
            palette='Set1',
            data=self.df,
            s=100
        )
        plt.title(f'Hierarchical Clustering (k={self.config["n_clusters"]})')
        plt.xlabel(self.config['plot_features'][0])
        plt.ylabel(self.config['plot_features'][1])
        plt.legend(title='Cluster')
        
        if save_fig and 'output_dir' in self.config:
            plt.savefig(os.path.join(self.config['output_dir'], 'clusters.png'), dpi=300)
        
        plt.show()
        return self
    
    def get_cluster_statistics(self):
        """
        Calculate and return statistics for each cluster.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing statistics for each cluster
        """
        # Create a DataFrame with only the features used for clustering and the cluster assignments
        features_with_clusters = self.df[self.config['feature_columns'] + ['Cluster']]
        
        # Calculate statistics only for the features used in clustering
        stats = features_with_clusters.groupby('Cluster').agg(['mean', 'std', 'min', 'max'])
        
        print("\nCluster Statistics (only for features used in clustering):")
        print(stats)
        return stats
    
    def save_results(self, filename='clustering_results.csv'):
        """
        Save the clustering results to a CSV file.
        
        Parameters:
        -----------
        filename : str, optional
            Name of the file to save results
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        if 'output_dir' in self.config:
            filepath = os.path.join(self.config['output_dir'], filename)
        else:
            filepath = filename
            
        self.df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        return self
    
    def generate_cluster_profiles(self):
        """
        Generate profiles for each cluster showing the mean values of all features.
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing cluster profiles with mean values for each feature
        """
        # Get cluster means for all features
        cluster_means = self.df.groupby('Cluster')[self.config['feature_columns']].mean()
        
        # Transpose the DataFrame to have clusters as columns
        profiles_df = cluster_means.T
        
        # Add size and percentage information
        cluster_sizes = self.df['Cluster'].value_counts().sort_index()
        cluster_percentages = (cluster_sizes / len(self.df) * 100).round(2)
        
        # Create a summary DataFrame for size and percentage
        summary_df = pd.DataFrame({
            'size': cluster_sizes,
            'percentage': cluster_percentages
        }).T
        
        # Combine the summary with the feature means
        final_profiles_df = pd.concat([summary_df, profiles_df])
        
        # Save to CSV
        if 'output_dir' in self.config:
            filepath = os.path.join(self.config['output_dir'], 'cluster_profiles.csv')
        else:
            filepath = 'cluster_profiles.csv'
            
        final_profiles_df.to_csv(filepath)
        print("\nCluster Profiles (Mean Values):")
        print(final_profiles_df)
        
        return final_profiles_df
    
    def run_full_analysis(self, save_outputs=False, use_pca_for_viz=True):
        """
        Run the complete hierarchical clustering analysis pipeline.
        
        Parameters:
        -----------
        save_outputs : bool, optional
            Whether to save outputs (plots and results)
        use_pca_for_viz : bool, optional
            Whether to use PCA for visualization
            
        Returns:
        --------
        self : HierarchicalClusteringAnalyzer
            Returns self for method chaining
        """
        self.preprocess_data()
        self.generate_linkage_matrix()
        self.plot_dendrogram(save_fig=save_outputs)
        self.perform_clustering()
        self.evaluate_clustering()
        self.generate_cluster_profiles()
        
        if use_pca_for_viz:
            self.apply_pca(n_components=2)
            self.visualize_clusters_pca(save_fig=save_outputs)
        else:
            self.visualize_clusters(save_fig=save_outputs)
            
        self.get_cluster_statistics()
        return self
    



# Example usage
if __name__ == "__main__":
    # Sample data creation (replace with your own data loading)
    try:
        data_path = '/teamspace/studios/this_studio/dataset/hb_outlier_removed.csv'
        print(f"Attempting to load data from: {data_path}")
        data = pd.read_csv(data_path)
        df = pd.DataFrame(data)
        df = df.drop(columns=[
            'country',
            'agent',
            'reservation_status_date',
            'reservation_status_Canceled',
            'reservation_status_No-Show',
            'reservation_status_Check-Out'
        ])
        df = df.sample(n=5000, random_state=42)
        
        # Configuration parameters - update these to match your actual column names
        config = {
            'feature_columns': [
                'lead_time',
                'adr',
                'required_car_parking_spaces',
                'total_of_special_requests',
                'is_city_hotel',
                'adults',
                'children',
                'is_repeated_guest',
                'is_canceled'
            ],
            'plot_features': ['lead_time', 'stays_in_weekend_nights'],
            'n_clusters': 7,
            'linkage_method': 'ward',
            'affinity': 'euclidean',
            'output_dir': '/teamspace/studios/this_studio/output'
        }
        
        # Create analyzer and run analysis
        analyzer = HierarchicalClusteringAnalyzer(config)
        analyzer.load_data(df).run_full_analysis(save_outputs=True)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Alternatively, you can run the analysis step by step:
    # analyzer = HierarchicalClusteringAnalyzer(config)
    # analyzer.load_data('your_dataset.csv')
    # analyzer.preprocess_data()
    # analyzer.generate_linkage_matrix()
    # analyzer.plot_dendrogram()
    # analyzer.perform_clustering()
    # analyzer.evaluate_clustering()
    # analyzer.visualize_clusters()
    # analyzer.get_cluster_statistics()
    # analyzer.save_results()
