import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from matplotlib.patches import Circle

EARTH_RADIUS = 3959.0  # miles


# ---------------------------------------------------------
# Helper: Haversine Distance
# ---------------------------------------------------------
def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS * 2 * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------
# STEP 1: Auto-select eps using KneeLocator
# ---------------------------------------------------------
def auto_detect_eps(df, min_samples=8, plot=True):
    coords = np.radians(df[['latitude', 'longitude']].values)

    nbrs = NearestNeighbors(n_neighbors=min_samples, metric='haversine').fit(coords)
    distances, _ = nbrs.kneighbors(coords)

    k_distances = np.sort(distances[:, min_samples - 1]) * EARTH_RADIUS  # convert to miles

    # Automatic elbow detection
    kneedle = KneeLocator(
        range(len(k_distances)),
        k_distances,
        curve='convex',
        direction='increasing',
        interp_method='polynomial'
    )
    eps_miles = k_distances[kneedle.knee] if kneedle.knee is not None else None

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(k_distances)
        if eps_miles:
            plt.axvline(kneedle.knee, color='red', linestyle='--', label='Detected Elbow')
        plt.title("K-Distance Graph (Auto eps)")
        plt.xlabel("Points")
        plt.ylabel(f"{min_samples}-NN Distance (miles)")
        plt.grid()
        plt.tight_layout()
        plt.legend()
        plt.show()

    return eps_miles


# ---------------------------------------------------------
# STEP 2: Run DBSCAN
# ---------------------------------------------------------
def run_dbscan(df, eps_miles, min_samples=8):
    eps = eps_miles / EARTH_RADIUS  # convert miles → radians
    coords = np.radians(df[['latitude', 'longitude']].values)

    db = DBSCAN(eps=eps, min_samples=min_samples, metric='haversine')
    labels = db.fit_predict(coords)

    df['geo_cluster'] = labels
    return df, labels


# ---------------------------------------------------------
# STEP 3: Evaluate clustering quality
# ---------------------------------------------------------
def evaluate_clustering(df, labels):
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters[unique_clusters != -1])

    print(f" Number of clusters (excluding noise): {n_clusters}")
    print("Cluster distribution:\n", pd.Series(labels).value_counts())

    # Silhouette score only on non-noise clusters
    valid = df[df['geo_cluster'] != -1]
    if len(valid.geo_cluster.unique()) > 1:
        coords = np.radians(valid[['latitude', 'longitude']].values)
        sil = silhouette_score(coords, valid['geo_cluster'], metric='haversine')
        print(f" Silhouette Score: {sil:.4f}")
    else:
        print(" Silhouette score not computed (only one cluster).")


# ---------------------------------------------------------
# STEP 4: Plot clusters with adaptive radius
# ---------------------------------------------------------
def plot_clusters(df):
    def draw_cluster_circle(ax, cluster_id, percentile=90):
        subset = df[df['geo_cluster'] == cluster_id]
        if subset.empty:
            return

        center_lat = subset['latitude'].mean()
        center_lon = subset['longitude'].mean()

        dists = haversine_np(
            center_lat, center_lon,
            subset['latitude'].values,
            subset['longitude'].values
        )
        radius_miles = np.percentile(dists, percentile)
        radius_deg = radius_miles / 69

        circle = Circle(
            (center_lon, center_lat),
            radius=radius_deg,
            edgecolor='darkblue',
            facecolor='none',
            linestyle='--',
            linewidth=2
        )
        ax.add_patch(circle)

    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    unique_clusters = sorted([c for c in df['geo_cluster'].unique() if c != -1])
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))

    for idx, cid in enumerate(unique_clusters):
        subset = df[df['geo_cluster'] == cid]
        ax.scatter(subset['longitude'], subset['latitude'],
                   s=30, color=colors[idx], alpha=0.7, label=f"Cluster {cid}")
        draw_cluster_circle(ax, cid)

    noise = df[df['geo_cluster'] == -1]
    ax.scatter(noise['longitude'], noise['latitude'], s=30, color='gray', alpha=0.4, label='Noise (-1)')

    plt.title("DBSCAN Geo Clusters with Adaptive Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# MASTER PIPELINE: Plug-and-Play Wrapper
# ---------------------------------------------------------
def geo_clustering_pipeline(merged_df, min_samples=8, user_eps=None):
    df = merged_df.dropna(subset=['latitude', 'longitude']).copy()

    #Auto-detect eps if user doesn't provide it
    if user_eps is None:
        print("\n Detecting EPS using KneeLocator...")
        eps_miles = auto_detect_eps(df, min_samples=min_samples)
        print(f" Selected eps: {eps_miles:.2f} miles")
    else:
        eps_miles = user_eps
        print(f" Using user-provided eps: {eps_miles} miles")

    #Run DBSCAN
    print("\n Running DBSCAN...")
    df, labels = run_dbscan(df, eps_miles, min_samples=min_samples)

    #Evaluate
    print("\n Clustering Evaluation:")
    evaluate_clustering(df, labels)

    #Plot
    print("\n Plotting Clusters...")
    plot_clusters(df)

    #Return merged back to main df
    return merged_df.merge(df[['store_id', 'geo_cluster']], on='store_id', how='left')

#Call the geo clustering pipeline - 37 eps value is with trial and error finalization and not necessarily automated
result_df = geo_clustering_pipeline(merged_df, min_samples=8, user_eps=37)
