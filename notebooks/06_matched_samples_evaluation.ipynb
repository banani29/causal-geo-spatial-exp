# Plot histograms for each covariate and the values should overlap with each other - Visual evidence
n_cols = 3
n_rows = int(np.ceil(len(covariates) / n_cols))
fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axs = axs.flatten()

for i, cov in enumerate(covariates):
    sns.histplot(
        data=merged_df,
        x=cov,
        hue='treated',
        element='step',
        stat='density',
        common_norm=False,
        palette={1: "steelblue", 0: "darkorange"},
        ax=axs[i]
    )
    axs[i].set_title(f'Distribution of {cov}')
    axs[i].legend(title="Treated", labels=["Control", "Treated"])

# Hide empty plots
for j in range(i + 1, len(axs)):
    axs[j].axis('off')

plt.tight_layout()
plt.suptitle("Covariate Distributions: Treated vs Control", fontsize=16, y=1.02)
plt.show()

### Calculate SMD to know the treatment and control units have SMD
# Ensure binary covariates are converted to numeric
categorical_cols = ['Family', 'Kids', 'Weekend', 'Holiday']
merged_df[categorical_cols] = merged_df[categorical_cols].replace({'Yes': 1, 'No': 0})

# Convert covariates to numeric in case of any remaining non-numeric entries
for cov in covariates:
    merged_df[cov] = pd.to_numeric(merged_df[cov], errors='coerce')

# Function to compute standardized mean difference
def compute_smd(treated_df, control_df, covariates):
    smd = {}
    for cov in covariates:
        treated_vals = pd.to_numeric(treated_df[cov], errors='coerce')
        control_vals = pd.to_numeric(control_df[cov], errors='coerce')
        mean_t = treated_vals.mean()
        mean_c = control_vals.mean()
        std_pooled = np.sqrt((treated_vals.std()**2 + control_vals.std()**2) / 2)
        smd[cov] = np.abs(mean_t - mean_c) / std_pooled if std_pooled != 0 else 0
    return pd.Series(smd).sort_values(ascending=False)

# Apply SMD calculation
smd_series = compute_smd(
    merged_df[merged_df['treated'] == 1],
    merged_df[merged_df['treated'] == 0],
    covariates
)

print(smd_series)
