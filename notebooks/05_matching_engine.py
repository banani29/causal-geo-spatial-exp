import numpy as np
import pandas as pd

# Parameters
SPATIAL_BUFFER_MILES = 25
EARTH_RADIUS = 3959.0
ALLOWED_CLUSTER_DIFF = 1
CONTROL_MATCHES_PER_TREATED = 1

# Haversine distance function
def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS * 2 * np.arcsin(np.sqrt(a))

# Step 1: Randomly assign 50% treatment
merged_df['treated'] = 0
treatment_indices = merged_df.sample(frac=0.5, random_state=42).index
merged_df.loc[treatment_indices, 'treated'] = 1

treated_df = merged_df[merged_df['treated'] == 1].copy()
control_df = merged_df[merged_df['treated'] == 0].copy()
used_control_ids = set()

matched_pairs = []

# Step 2: Match treated with control
for t_idx, t_row in treated_df.iterrows():
    t_lat, t_lon = t_row['latitude'], t_row['longitude']
    t_kmeans = t_row['kmeans_cluster']
    t_geo = t_row['geo_cluster']

    # Filter candidate controls
    candidates = control_df[
        (control_df['kmeans_cluster'] - t_kmeans).abs() <= ALLOWED_CLUSTER_DIFF &
        (~control_df.index.isin(used_control_ids))  # prevent reuse
    ].copy()

    if candidates.empty:
        continue

    # Compute Haversine distance
    distances = haversine_np(
        t_lat, t_lon,
        candidates['latitude'].values,
        candidates['longitude'].values
    )

    # Enforce spatial constraint if same geo cluster
    spatial_filter = (
        (candidates['geo_cluster'] != t_geo) |
        ((candidates['geo_cluster'] == t_geo) & (distances >= SPATIAL_BUFFER_MILES))
    )

    valid_controls = candidates[spatial_filter]

    if valid_controls.empty:
        continue

    sampled_controls = valid_controls.sample(
        n=min(CONTROL_MATCHES_PER_TREATED, len(valid_controls)),
        random_state=42
    )

    for _, c_row in sampled_controls.iterrows():
        matched_pairs.append({
            'treated_id': t_idx,
            'control_id': c_row.name
        })
        used_control_ids.add(c_row.name)

# Step 3: Combine matched rows
treated_final = treated_df.loc[[p['treated_id'] for p in matched_pairs]].copy()
control_final = control_df.loc[[p['control_id'] for p in matched_pairs]].copy()

treated_final['group'] = 'treated'
control_final['group'] = 'control'
final_df = pd.concat([treated_final, control_final], ignore_index=True)

print(f" Matched {len(treated_final)} treated with {len(control_final)} controls.")

