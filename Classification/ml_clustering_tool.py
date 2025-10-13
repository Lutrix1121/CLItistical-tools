import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, BisectingKMeans
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score, adjusted_rand_score)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import argparse
import warnings
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

AVAILABLE_ALGORITHMS = {
    'kmeans': KMeans,
    'kmedoids': KMedoids,
    'agglomerative': AgglomerativeClustering,
    'divisive': BisectingKMeans,
    'dbscan': DBSCAN
}

def get_model_params(algorithm, n_clusters=3, random_state=42, **kwargs):
    """Get default parameters for each algorithm."""
    params = {
        'kmeans': {
            'n_clusters': n_clusters,
            'random_state': random_state,
            'n_init': 10,
            'max_iter': 300
        },
        'kmedoids': {
            'n_clusters': n_clusters,
            'random_state': random_state,
            'max_iter': 300
        },
        'agglomerative': {
            'n_clusters': n_clusters,
            'linkage': kwargs.get('linkage', 'ward')
        },
        'divisive': {
            'n_clusters': n_clusters,
            'random_state': random_state,
            'n_init': 10,
            'max_iter': 300,
            'bisecting_strategy': kwargs.get('bisecting_strategy', 'biggest_inertia')
        },
        'dbscan': {
            'eps': kwargs.get('eps', 0.5),
            'min_samples': kwargs.get('min_samples', 5)
        }
    }
    return params.get(algorithm, {})

def perform_clustering(csv_file, algorithm='kmeans', n_clusters=3,
                      feature_columns=None, standardize=True, separator=',',
                      random_state=42, eps=0.5, min_samples=5, 
                      linkage_method='ward', bisecting_strategy='biggest_inertia',
                      elbow_analysis=False, silhouette_analysis=False, 
                      visualize=False, output_clustered_data=None):
    """
    Performs clustering analysis on data from a CSV file using various algorithms.
    
    Parameters:
    csv_file (str): Path to the CSV file.
    algorithm (str): Clustering algorithm to use.
    n_clusters (int): Number of clusters (not used for DBSCAN).
    feature_columns (list, optional): List of feature column names.
    standardize (bool): Whether to standardize features (default: True).
    separator (str): Separator used in CSV file (default: ',').
    random_state (int): Random state for reproducibility (default: 42).
    eps (float): DBSCAN epsilon parameter (default: 0.5).
    min_samples (int): DBSCAN min_samples parameter (default: 5).
    linkage_method (str): Linkage method for hierarchical clustering (default: 'ward').
    bisecting_strategy (str): Strategy for divisive clustering (default: 'biggest_inertia').
    elbow_analysis (bool): Perform elbow method analysis (default: False).
    silhouette_analysis (bool): Perform silhouette analysis (default: False).
    visualize (bool): Create visualization plots (default: False).
    output_clustered_data (str): Path to save clustered data (optional).
    
    Returns:
    dict: Results containing clustering metrics and statistics.
    """
    try:
        data = pd.read_csv(csv_file, sep=separator)
        print(f"Loaded data with shape: {data.shape}")
        
        # Select feature columns
        if feature_columns is None:
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_columns = numeric_columns
            print(f"Using all numeric columns as features: {feature_columns}")
        else:
            missing_columns = [col for col in feature_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Feature columns do not exist in dataset: {missing_columns}")
        
        if not feature_columns:
            raise ValueError("No feature columns available for analysis.")
        
        X = data[feature_columns].copy()
        
        # Handle missing values
        if X.isnull().any().any():
            missing_info = X.isnull().sum()
            missing_cols = missing_info[missing_info > 0]
            print(f"Warning: Found missing values in feature columns:\n{missing_cols}")
            print("Removing rows with missing values.")
            X = X.dropna()
            data = data.loc[X.index]
        
        # Handle categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        if categorical_columns:
            print(f"Encoding categorical columns: {categorical_columns}")
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
            feature_columns = X.columns.tolist()
        
        if len(X) < 10:
            raise ValueError("Insufficient data for analysis. Need at least 10 observations after cleaning.")
        
        print(f"Final dataset shape: {X.shape}")
        
        # Standardize features if requested
        scaler = None
        if standardize:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Perform elbow analysis if requested
        elbow_results = None
        if elbow_analysis and algorithm in ['kmeans', 'kmedoids', 'divisive']:
            print("\nPerforming elbow analysis...")
            elbow_results = perform_elbow_analysis(X_scaled, algorithm, random_state)
        
        # Perform silhouette analysis if requested
        silhouette_results = None
        if silhouette_analysis and algorithm in ['kmeans', 'kmedoids', 'agglomerative', 'divisive']:
            print("\nPerforming silhouette analysis...")
            silhouette_results = perform_silhouette_analysis(X_scaled, algorithm, random_state, linkage_method, bisecting_strategy)
        
        # Train clustering model
        print(f"\n{'='*60}")
        print(f"Training {algorithm.upper().replace('_', ' ')}...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        model_class = AVAILABLE_ALGORITHMS[algorithm]
        
        if algorithm == 'dbscan':
            model_params = get_model_params(algorithm, eps=eps, min_samples=min_samples)
        elif algorithm == 'agglomerative':
            model_params = get_model_params(algorithm, n_clusters=n_clusters, linkage=linkage_method)
        else:
            model_params = get_model_params(algorithm, n_clusters=n_clusters, random_state=random_state)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = model_class(**model_params)
            labels = model.fit_predict(X_scaled)
        
        training_time = time.time() - start_time
        
        # Get unique clusters
        unique_labels = np.unique(labels)
        n_clusters_found = len(unique_labels[unique_labels != -1])  
	# Exclude noise for DBSCAN
        n_noise = np.sum(labels == -1) if algorithm == 'dbscan' else 0
        
        print(f"Number of clusters found: {n_clusters_found}")
        if n_noise > 0:
            print(f"Number of noise points: {n_noise}")
        
        # Calculate clustering metrics
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        
        # Only calculate metrics if we have at least 2 clusters and some valid assignments
        valid_labels = labels[labels != -1] if algorithm == 'dbscan' else labels
        X_valid = X_scaled[labels != -1] if algorithm == 'dbscan' else X_scaled
        
        silhouette = None
        calinski = None
        davies_bouldin = None
        
        if n_clusters_found >= 2 and len(valid_labels) >= 2:
            try:
                silhouette = silhouette_score(X_valid, valid_labels)
                calinski = calinski_harabasz_score(X_valid, valid_labels)
                davies_bouldin = davies_bouldin_score(X_valid, valid_labels)
            except Exception as e:
                print(f"Warning: Could not calculate some metrics: {str(e)}")
        
        # Calculate inertia for K-Means
        inertia = None
        if hasattr(model, 'inertia_'):
            inertia = model.inertia_
        
        # Get cluster centers
        cluster_centers = None
        if hasattr(model, 'cluster_centers_'):
            cluster_centers = model.cluster_centers_
        elif hasattr(model, 'medoid_indices_'):
            cluster_centers = X_scaled[model.medoid_indices_]
        
        # Calculate cluster statistics
        cluster_stats = []
        for label in unique_labels:
            if label == -1:  # Skip noise for DBSCAN
                continue
            cluster_mask = labels == label
            cluster_data = X_scaled[cluster_mask]
            
            stats = {
                'cluster': label,
                'size': np.sum(cluster_mask),
                'percentage': np.sum(cluster_mask) / len(labels) * 100,
                'mean': np.mean(cluster_data, axis=0),
                'std': np.std(cluster_data, axis=0)
            }
            cluster_stats.append(stats)
        
        # Add cluster labels to original data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = labels
        
        # Save clustered data if requested
        if output_clustered_data:
            data_with_clusters.to_csv(output_clustered_data, index=False)
            print(f"Clustered data saved to: {output_clustered_data}")
        
        # Create visualizations if requested
        visualization_files = None
        if visualize:
            print("\nGenerating visualizations...")
            visualization_files = create_visualizations(
                X_scaled, labels, feature_columns, algorithm, 
                cluster_centers, csv_file
            )
        
        results = {
            'algorithm': algorithm,
            'model': model,
            'scaler': scaler,
            'feature_columns': feature_columns,
            'n_clusters': n_clusters_found,
            'n_noise': n_noise,
            'labels': labels,
            'cluster_sizes': cluster_sizes.to_dict(),
            'cluster_stats': cluster_stats,
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'inertia': inertia,
            'cluster_centers': cluster_centers,
            'n_samples': len(X),
            'n_features': X_scaled.shape[1],
            'standardized': standardize,
            'training_time': training_time,
            'elbow_results': elbow_results,
            'silhouette_results': silhouette_results,
            'visualization_files': visualization_files,
            'data_with_clusters': data_with_clusters
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def perform_elbow_analysis(X, algorithm, random_state, max_clusters=10):
    """Perform elbow method analysis to find optimal number of clusters."""
    K_range = range(2, min(max_clusters + 1, len(X)))
    inertias = []
    
    for k in K_range:
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        else:  # kmedoids
            model = KMedoids(n_clusters=k, random_state=random_state)
        
        model.fit(X)
        inertias.append(model.inertia_)
    
    return {'k_values': list(K_range), 'inertias': inertias}

def perform_silhouette_analysis(X, algorithm, random_state, linkage_method, max_clusters=10):
    """Perform silhouette analysis to find optimal number of clusters."""
    K_range = range(2, min(max_clusters + 1, len(X)))
    silhouette_scores = []
    
    for k in K_range:
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        elif algorithm == 'kmedoids':
            model = KMedoids(n_clusters=k, random_state=random_state)
        else:  # agglomerative
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
        
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)
    
    return {'k_values': list(K_range), 'silhouette_scores': silhouette_scores}

def create_visualizations(X, labels, feature_names, algorithm, cluster_centers, csv_file):
    """Create visualization plots for clustering results."""
    import os
    
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = f"{base_name}_{algorithm}_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    visualization_files = []
    
    # 1. PCA visualization (2D)
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        explained_var = pca.explained_variance_ratio_
    else:
        X_pca = X
        explained_var = [1.0, 1.0]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    
    if cluster_centers is not None:
        if X.shape[1] > 2:
            centers_pca = pca.transform(cluster_centers)
        else:
            centers_pca = cluster_centers
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', 
                   marker='X', s=300, edgecolors='black', linewidth=2, 
                   label='Cluster Centers')
    
    plt.xlabel(f'PC1 ({explained_var[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({explained_var[1]:.2%} variance)')
    plt.title(f'Clustering Results - {algorithm.upper()}')
    plt.colorbar(scatter, label='Cluster')
    if cluster_centers is not None:
        plt.legend()
    plt.tight_layout()
    
    pca_file = os.path.join(output_dir, 'pca_visualization.png')
    plt.savefig(pca_file, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_files.append(pca_file)
    
    # 2. Cluster size distribution
    unique_labels = np.unique(labels)
    cluster_sizes = [np.sum(labels == label) for label in unique_labels]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(unique_labels)), cluster_sizes, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.xticks(range(len(unique_labels)), [f'{label}' for label in unique_labels])
    
    for bar, size in zip(bars, cluster_sizes):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{size}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    dist_file = os.path.join(output_dir, 'cluster_distribution.png')
    plt.savefig(dist_file, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_files.append(dist_file)
    
    # 3. Feature correlation heatmap (first cluster vs all)
    if len(feature_names) <= 20:  # Only for reasonable number of features
        plt.figure(figsize=(12, 10))
        
        # Calculate mean feature values per cluster
        n_clusters = len([l for l in unique_labels if l != -1])
        if n_clusters > 0:
            cluster_means = np.zeros((n_clusters, X.shape[1]))
            valid_labels = [l for l in unique_labels if l != -1]
            
            for i, label in enumerate(valid_labels):
                cluster_mask = labels == label
                cluster_means[i, :] = np.mean(X[cluster_mask], axis=0)
            
            df_means = pd.DataFrame(cluster_means.T, 
                                   index=feature_names, 
                                   columns=[f'Cluster {l}' for l in valid_labels])
            
            sns.heatmap(df_means, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, cbar_kws={'label': 'Standardized Value'})
            plt.title('Mean Feature Values by Cluster')
            plt.tight_layout()
            
            heatmap_file = os.path.join(output_dir, 'feature_heatmap.png')
            plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_files.append(heatmap_file)
    
    return visualization_files

def print_results(results, output_file=None):
    """Print formatted results of clustering analysis."""
    if results is None:
        return
    
    original_stdout = None
    if output_file is not None:
        import sys
        import os
        
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        original_stdout = sys.stdout
        sys.stdout = open(output_file, 'w', encoding='utf-8')
    
    print(f"\n{results['algorithm'].upper().replace('_', ' ')} - CLUSTERING RESULTS")
    print("=" * 100)
    print(f"Number of samples: {results['n_samples']}")
    print(f"Number of features: {results['n_features']}")
    print(f"Number of clusters: {results['n_clusters']}")
    if results['n_noise'] > 0:
        print(f"Number of noise points: {results['n_noise']}")
    print(f"Features standardized: {results['standardized']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print("=" * 100)
    
    print("\nCLUSTERING METRICS:")
    print("-" * 50)
    if results['silhouette_score'] is not None:
        print(f"Silhouette Score:        {results['silhouette_score']:.4f}")
        print(f"  (Range: -1 to 1, higher is better)")
    if results['calinski_harabasz_score'] is not None:
        print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']:.4f}")
        print(f"  (Higher is better)")
    if results['davies_bouldin_score'] is not None:
        print(f"Davies-Bouldin Score:    {results['davies_bouldin_score']:.4f}")
        print(f"  (Lower is better)")
    if results['inertia'] is not None:
        print(f"Inertia (Within-Cluster Sum of Squares): {results['inertia']:.4f}")
    
    print("\nCLUSTER DISTRIBUTION:")
    print("-" * 50)
    print(f"{'Cluster':<10} {'Size':<10} {'Percentage':<12}")
    print("-" * 50)
    
    for cluster_id, size in results['cluster_sizes'].items():
        percentage = (size / results['n_samples']) * 100
        cluster_label = 'Noise' if cluster_id == -1 else f'{cluster_id}'
        print(f"{cluster_label:<10} {size:<10} {percentage:<12.2f}%")
    
    print("\nCLUSTER STATISTICS:")
    print("-" * 100)
    
    for stats in results['cluster_stats']:
        print(f"\nCluster {stats['cluster']}:")
        print(f"  Size: {stats['size']} ({stats['percentage']:.2f}%)")
        print(f"  Mean values (top 5 features):")
        
        # Show top 5 features by absolute mean value
        feature_means = list(zip(results['feature_columns'], stats['mean']))
        feature_means_sorted = sorted(feature_means, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for feature, mean_val in feature_means_sorted:
            print(f"    {feature[:40]:<40}: {mean_val:>10.4f}")
    
    if results['elbow_results'] is not None:
        print("\nELBOW ANALYSIS RESULTS:")
        print("-" * 50)
        print(f"{'K':<10} {'Inertia':<15}")
        print("-" * 50)
        for k, inertia in zip(results['elbow_results']['k_values'], 
                             results['elbow_results']['inertias']):
            print(f"{k:<10} {inertia:<15.4f}")
    
    if results['silhouette_results'] is not None:
        print("\nSILHOUETTE ANALYSIS RESULTS:")
        print("-" * 50)
        print(f"{'K':<10} {'Silhouette Score':<20}")
        print("-" * 50)
        for k, score in zip(results['silhouette_results']['k_values'], 
                           results['silhouette_results']['silhouette_scores']):
            print(f"{k:<10} {score:<20.4f}")
        
        # Find optimal K
        best_idx = np.argmax(results['silhouette_results']['silhouette_scores'])
        best_k = results['silhouette_results']['k_values'][best_idx]
        best_score = results['silhouette_results']['silhouette_scores'][best_idx]
        print(f"\nOptimal K (by silhouette): {best_k} (score: {best_score:.4f})")
    
    print("\nCLUSTERING INTERPRETATION:")
    print("-" * 50)
    
    if results['silhouette_score'] is not None:
        if results['silhouette_score'] > 0.7:
            quality = "excellent - strong cluster structure"
        elif results['silhouette_score'] > 0.5:
            quality = "good - reasonable cluster structure"
        elif results['silhouette_score'] > 0.25:
            quality = "moderate - weak cluster structure"
        else:
            quality = "poor - overlapping clusters"
        
        print(f"The clustering quality is {quality}.")
    
    # Check for cluster balance
    sizes = list(results['cluster_sizes'].values())
    if -1 in results['cluster_sizes']:
        sizes = [s for c, s in results['cluster_sizes'].items() if c != -1]
    
    if sizes:
        max_size = max(sizes)
        min_size = min(sizes)
        if max_size > 3 * min_size:
            print("\nWarning: Unbalanced cluster sizes detected.")
            print("Consider:")
            print("- Adjusting the number of clusters")
            print("- Trying different algorithms")
            print("- Feature scaling or transformation")
    
    if results['n_clusters'] < 2 and results['algorithm'] != 'dbscan':
        print("\nWarning: Only one cluster found. Data may not have natural groupings.")
    
    if results['visualization_files']:
        print("\nVISUALIZATIONS CREATED:")
        print("-" * 50)
        for vfile in results['visualization_files']:
            print(f"  {vfile}")
    
    if original_stdout is not None:
        sys.stdout.close()
        sys.stdout = original_stdout
        print(f"Results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Machine Learning Clustering Tool - Supports multiple algorithms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Algorithms:
  kmeans          - K-Means Clustering (partitioning)
  kmedoids        - K-Medoids Clustering (partitioning)
  agglomerative   - Agglomerative Hierarchical Clustering
  dbscan          - DBSCAN (Density-Based Clustering)

Examples:
  python ml_clustering_tool.py data.csv --algorithm kmeans --n_clusters 3
  python ml_clustering_tool.py data.csv --algorithm dbscan --eps 0.5 --min_samples 5
  python ml_clustering_tool.py data.csv --algorithm agglomerative --n_clusters 4 --linkage complete
  python ml_clustering_tool.py data.csv --algorithm kmeans --elbow_analysis --silhouette_analysis
  python ml_clustering_tool.py data.csv --algorithm kmeans --n_clusters 3 --visualize --output results.txt
        """
    )
    
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--algorithm', type=str, default='kmeans',
                        choices=['kmeans', 'kmedoids', 'agglomerative', 'dbscan'],
                        help='Clustering algorithm to use (default: kmeans)')
    parser.add_argument('--n_clusters', type=int, default=3,
                        help='Number of clusters (not used for DBSCAN) (default: 3)')
    parser.add_argument('--feature_columns', type=str, nargs='+', default=None,
                        help='Names of feature columns (if not specified, uses all numeric columns)')
    parser.add_argument('--no_standardize', action='store_true',
                        help='Do not standardize features (default: standardize)')
    parser.add_argument('--separator', type=str, default=',',
                        help='Separator used in CSV file (default: ",")')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random state for reproducibility (default: 42)')
    parser.add_argument('--eps', type=float, default=0.5,
                        help='DBSCAN epsilon parameter (default: 0.5)')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='DBSCAN min_samples parameter (default: 5)')
    parser.add_argument('--linkage', type=str, default='ward',
                        choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage method for hierarchical clustering (default: ward)')
    parser.add_argument('--elbow_analysis', action='store_true',
                        help='Perform elbow method analysis (for kmeans/kmedoids)')
    parser.add_argument('--silhouette_analysis', action='store_true',
                        help='Perform silhouette analysis')
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path to save results (optional)')
    parser.add_argument('--output_data', type=str, default=None,
                        help='Output file path to save clustered data (optional)')
    
    args = parser.parse_args()
    
    results = perform_clustering(
        args.csv_file, args.algorithm, args.n_clusters,
        args.feature_columns, not args.no_standardize, args.separator,
        args.random_state, args.eps, args.min_samples, args.linkage,
        args.elbow_analysis, args.silhouette_analysis, args.visualize,
        args.output_data
    )
    
    print_results(results, args.output)

if __name__ == "__main__":
    main()
