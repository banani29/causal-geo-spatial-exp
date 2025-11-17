#Determine the right SPATIAL_BUFFER_MILES
EARTH_RADIUS = 3959.0
ALLOWED_CLUSTER_DIFF = 1
CONTROL_MATCHES_PER_TREATED = 1
SPATIAL_BUFFER_OPTIONS = list(range(0, 51, 5))  # 0 to 50 miles in steps of 5

# Haversine function
def haversine_np(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return EARTH_RADIUS * 2 * np.arcsin(np.sqrt(a))

results = []

# Iterate over buffer values
for buffer in SPATIAL_BUFFER_OPTIONS:
    merged_df['treated'] = 0
    treatment_indices = merged_df.sample(frac=0.5, random_state=42).index
    merged_df.loc[treatment_indices, 'treated'] = 1

    treated_df = merged_df[merged_df['treated'] == 1].copy()
    control_df = merged_df[merged_df['treated'] == 0].copy()
    used_control_ids = set()
    matched_pairs = []

    # For SMD calculation
    matched_covariates = []

    for t_idx, t_row in treated_df.iterrows():
        t_lat, t_lon = t_row['latitude'], t_row['longitude']
        t_kmeans = t_row['kmeans_cluster']
        t_geo = t_row['geo_cluster']

        candidates = control_df[
            (control_df['kmeans_cluster'] - t_kmeans).abs() <= ALLOWED_CLUSTER_DIFF &
            (~control_df.index.isin(used_control_ids))
        ].copy()

        if candidates.empty:
            continue

        distances = haversine_np(
            t_lat, t_lon,
            candidates['latitude'].values,
            candidates['longitude'].values
        )

        spatial_filter = (
            (candidates['geo_cluster'] != t_geo) |
            ((candidates['geo_cluster'] == t_geo) & (distances >= buffer))
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
            matched_covariates.append({
                'buffer': buffer,
                'Store_Size_treated': t_row['Store_Size'],
                'Foot_Traffic_treated': t_row['Foot_Traffic'],
                'Inventory_Level_treated': t_row['Inventory_Level'],
                'Store_Size_control': c_row['Store_Size'],
                'Foot_Traffic_control': c_row['Foot_Traffic'],
                'Inventory_Level_control': c_row['Inventory_Level']
            })

    # SMD Calculation
    if matched_covariates:
        matched_df = pd.DataFrame(matched_covariates)
        smds = {}
        for cov in ['Store_Size', 'Foot_Traffic', 'Inventory_Level']:
            treated_mean = matched_df[f'{cov}_treated'].mean()
            control_mean = matched_df[f'{cov}_control'].mean()
            pooled_sd = np.sqrt((matched_df[f'{cov}_treated'].std()**2 + matched_df[f'{cov}_control'].std()**2) / 2)
            smd = abs(treated_mean - control_mean) / pooled_sd
            smds[cov] = smd
    else:
        smds = {'Store_Size': np.nan, 'Foot_Traffic': np.nan, 'Inventory_Level': np.nan}

    results.append({
        'Spatial_Buffer_Miles': buffer,
        'Matched_Pairs': len(matched_pairs),
        'SMD_Store_Size': smds['Store_Size'],
        'SMD_Foot_Traffic': smds['Foot_Traffic'],
        'SMD_Inventory_Level': smds['Inventory_Level'],
        'Duplicated_Controls': len(matched_pairs) - len(used_control_ids)
    })

results_df = pd.DataFrame(results)
print(results_df)

# Plot 1: SMDs vs Spatial Buffer
plt.figure(figsize=(10, 6))
plt.plot(results_df['Spatial_Buffer_Miles'], results_df['SMD_Store_Size'], label='SMD Store Size', marker='o')
plt.plot(results_df['Spatial_Buffer_Miles'], results_df['SMD_Foot_Traffic'], label='SMD Foot Traffic', marker='s')
plt.plot(results_df['Spatial_Buffer_Miles'], results_df['SMD_Inventory_Level'], label='SMD Inventory Level', marker='^')
plt.axvline(x=25, color='gray', linestyle='--', label='Chosen Threshold (25 miles)')
plt.title('Standardized Mean Differences vs Spatial Buffer')
plt.xlabel('Spatial Buffer (Miles)')
plt.ylabel('SMD')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Duplicated Controls vs Spatial Buffer
plt.figure(figsize=(10, 5))
plt.plot(results_df['Spatial_Buffer_Miles'], results_df['Duplicated_Controls'], label='Duplicated Controls', color='darkred', marker='o')
plt.axvline(x=25, color='gray', linestyle='--', label='Chosen Threshold (25 miles)')
plt.title('Duplicated Control Units vs Spatial Buffer')
plt.xlabel('Spatial Buffer (Miles)')
plt.ylabel('Number of Duplicated Controls')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
