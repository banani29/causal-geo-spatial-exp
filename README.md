**A Geo-Causal and Risk-Aware Framework for Evaluating Localized Interventions in Spatially Distributed Systems**

**Contributors : Banani Mohapatra, Bhavnish Walia, Manisha Arora **

A modular pipeline for geospatial clustering, risk-aware matching, spillover detection, and causal inference — designed for retail stores, and extensible to any geographically distributed system such as ride-sharing, delivery networks, urban mobility, and public policy pilots.

This repository provides an end-to-end, plug-and-play causal inference workflow to evaluate the impact of localized interventions where both geography and operational heterogeneity matter. It is particularly well-suited for retail store interventions, where spatial proximity, neighborhood-level effects, and store-level covariates strongly influence outcomes.

But the same framework generalizes cleanly to other spatial systems such as:
	•	Ride-sharing zones and driver incentive experiments
	•	Local delivery networks (DoorDash / Instacart–style geos)
	•	Urban mobility pilots (e-bikes, express lanes, microtransit)
	•	Public policy and neighborhood-level interventions
	•	Healthcare service-area rollouts

Wherever units are located in space and treatment spillovers are possible, this framework applies.

**Repository Structure**

├── 01_data_loading_and_cleaning.ipynb
├── 02_geo_clustering.ipynb
├── 03_covariate_clustering.ipynb
├── 04_determine_spatial_buffer.ipynb
├── 05_matching_engine.ipynb
├── 06_matched_samples_evaluation.ipynb
├── 07_did_ipw_pipeline.ipynb
└── 08_causal_pipeline_results_robustness_checks.ipynb

** Notebook Overview**

1️⃣ 01_data_loading_and_cleaning.ipynb

Import, inspect, and merge store datasets with daily/weekly retail metrics. (Dataset folder contains Store_Dataset.csv/ store_location_dataset.csv / FreshRetailNet-50K is imported from python "datasets" library)


2️⃣ 02_geo_clustering.ipynb

Perform geospatial clustering using:
	•	DBSCAN with haversine distance
	•	Auto-detected EPS via KneeLocator
	•	Visualization of clusters + noise
	•	Adaptive-radius cluster boundaries

3️⃣ 03_covariate_clustering.ipynb

Cluster stores by operational or demographic covariates:
	•	Store size
	•	Foot traffic
	•	Inventory
	•	Customer profiles
	•	Online/offline demand
	•	Staffing

Uses PCA + KMeans (Elbow method) + silhouette scoring.

4️⃣ 04_determine_spatial_buffer.ipynb

Systematically determine an optimal spatial separation buffer to reduce spillovers.

Analyzes:
	•	Duplicate controls
	•	Match feasibility
	•	Covariate balance (SMD)
	•	Buffer ↔ sample-size trade-offs

5️⃣ 05_matching_engine.ipynb

Geo-aware + covariate-aware matching engine enforcing:
	•	Cluster similarity
	•	Spatial buffer rules
	•	Single-use controls
	•	Haversine-based nearest neighbors
	•	Risk-aware logic to avoid high-spillover geos

6️⃣ 06_matched_samples_evaluation.ipynb

Evaluate matched treated/control pairs:
	•	Covariate balance plots
	•	Distribution comparisons
	•	Standardized Mean Differences (SMD)

7️⃣ 07_did_ipw_pipeline.ipynb

Estimate causal impact using:
	•	Logistic regression propensity scores
	•	Inverse Propensity Weights (IPW)
	•	Weighted Difference-in-Differences (DiD)
	•	Clustered standard errors
	•	ATE + confidence intervals
	•	IPW distribution diagnostics

8️⃣ 08_causal_pipeline_results_robustness_checks.ipynb

Perform robustness diagnostics:
	•	Pre-trend checks
	•	Placebo tests
	•	Residual diagnostics
	•	Event Study regressions
  •	Covariate adjustments

**Primary Domain: Retail Store Interventions**

This framework is optimized for retail settings, where:
	•	Stores cluster geographically
	•	Customers travel across catchments
	•	Spillover effects distort naive A/B testing
	•	Local market characteristics matter

