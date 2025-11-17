# Create the propensity score of being included as a part of treatment using logistic regression
# -------------------------------------------------
# STEP 1: Propensity score generation
# -------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

covariates = [
    'Discount_Rate', 'Customer_Age', 'Store_Size', 'Inventory_Level',
    'Number_of_Employees', 'Marketing_Spend', 'Family', 'Kids', 'Weekend',
    'Holiday', 'Foot_Traffic', 'Average_Transaction_Value', 'Online_Sales'
]

X = final_df[covariates].apply(pd.to_numeric, errors='coerce').fillna(0)
y = final_df['treated']
X_scaled = StandardScaler().fit_transform(X)

model = LogisticRegression()
model.fit(X_scaled, y)

final_df['propensity_score'] = model.predict_proba(X_scaled)[:, 1]


# -------------------------------------------------
# STEP 2: Long Format Panel for DiD
# -------------------------------------------------
long_df = pd.concat([
    pd.DataFrame({
        'store_id': final_df['store_id'],
        'sales': final_df['pre_sales'],
        'treated': final_df['treated'],
        'post': 0,
        'propensity_score': final_df['propensity_score']
    }),
    pd.DataFrame({
        'store_id': final_df['store_id'],
        'sales': final_df['post_sales'],
        'treated': final_df['treated'],
        'post': 1,
        'propensity_score': final_df['propensity_score']
    })
], ignore_index=True)

# -------------------------------------------------
# STEP 3: Compute IPW Weights
# -------------------------------------------------
def compute_ipw(row):
    p = np.clip(row['propensity_score'], 1e-3, 1 - 1e-3)  # avoid div by 0
    return 1 / p if row['treated'] == 1 else 1 / (1 - p)

long_df['ipw'] = long_df.apply(compute_ipw, axis=1)

# -------------------------------------------------
# STEP 4: Weighted DiD Regression with Clustered SE
# -------------------------------------------------
did_model = smf.wls("sales ~ treated * post", data=long_df, weights=long_df['ipw']).fit(
    cov_type='cluster',
    cov_kwds={'groups': long_df['store_id']}
)

print("\nDiD + IPW Regression Summary (Clustered SE):\n")
print(did_model.summary())

# -------------------------------------------------
# STEP 5: Estimate ATE
# -------------------------------------------------
ate_ipw = did_model.params['treated:post']
ate_se = did_model.bse['treated:post']
ate_ci = (ate_ipw - 1.96 * ate_se, ate_ipw + 1.96 * ate_se)

print(f"\nEstimated ATE (IPW + DiD): {ate_ipw:.4f}")
print(f"95% CI: [{ate_ci[0]:.4f}, {ate_ci[1]:.4f}]")

# -------------------------------------------------
# STEP 6: Visualize IPW Distribution
# -------------------------------------------------
plt.figure(figsize=(8, 4))
sns.histplot(data=long_df, x='ipw', hue='treated', bins=30, element='step', common_norm=False)
plt.title('Distribution of Inverse Propensity Weights')
plt.xlabel('IPW')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------------------------
# STEP 7: Plot Group-level Mean Sales (Optional)
# -------------------------------------------------
group_summary = (
    long_df
    .groupby(['treated', 'post'])['sales']
    .mean()
    .unstack()
    .rename(index={0: 'Control', 1: 'Treated'}, columns={0: 'Pre', 1: 'Post'})
)

group_summary.plot(kind='bar', figsize=(6, 4), rot=0)
plt.title('Group-Level Mean Sales: Pre vs Post')
plt.ylabel('Avg Sales')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


