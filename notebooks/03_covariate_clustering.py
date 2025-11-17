import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# -------------------------
# STEP 1: Covariate Setup
# -------------------------
covariates = [
    'Discount_Rate', 'Customer_Age', 'Store_Size', 'Inventory_Level',
    'Number_of_Employees', 'Marketing_Spend', 'Family', 'Kids', 'Weekend',
    'Holiday', 'Foot_Traffic', 'Average_Transaction_Value', 'Online_Sales'
]

# -------------------------
# STEP 2: Preprocess
# -------------------------
X = merged_df[covariates].apply(pd.to_numeric, errors='coerce').fillna(0)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------
# STEP 3: PCA Reduction
# -------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# -------------------------
# STEP 4: Elbow Curve on PCA
# -------------------------
sse = []
k_range = range(2, 15)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init='auto')
    km.fit(X_pca)
    sse.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('SSE (Inertia)')
plt.title('Elbow Curve to Determine Optimal k (PCA)')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# STEP 5: KMeans Clustering (PCA space)
# -------------------------
optimal_k = 4  # Adjust based on elbow
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
labels = kmeans.fit_predict(X_pca)

merged_df['kmeans_cluster'] = labels
silhouette = silhouette_score(X_pca, labels)
print(f"Silhouette Score after PCA (k={optimal_k}): {silhouette:.4f}")

# -------------------------
# STEP 6: PCA Scatterplot Visualization
# -------------------------
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set2', s=50)
plt.title(f'PCA Projection of Store Clusters (k={optimal_k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()
