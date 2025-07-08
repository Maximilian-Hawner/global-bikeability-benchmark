"""
Global Bikeability Benchmarking – clustering.py
========================================================
Generates clustering for bikeability, connectivity
and accessibility dimensions using the values calculated in workflow.py.
Clustering method described in:

M. Hawner et al. (2025): "Global Bikeability Benchmarking:
Comparative Analysis of 100 Cities Worldwide Using NetAScore",
Journal of Cycling & Micromobility Research. DOI: 

Author: Maximilian Hawner, University of Salzburg  
Contact:   
Version:   1.0 – 2025-06-19
License:   

Usage
-----
edit the following:
input_csv   = file path to result-csv of workflow.py
output_folder = folder path to preferred location of clustering output
dendro_configs = configuration for the preferred number of clusters for each clustering

Requirements
------------
Python ≥3.9, others: see import

This script expects the CSV with calculation results from workflow.py located in `input_csv`
and produces clustering on bikeability, connectivity, accessibility and all together. Method is described in Section 2.5
of the paper. The results are then stored as 
    - dendograms (Ward Linkage with Euclidean Distance),
    - CSV (Cities and cluster-membership), 
    - and txt-file (Silhouette Score and CH Index). 

Copyright © 2025 Maximilian Hawner
"""

# import of libraries
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Setup paths
input_csv   = r"C:\Users\Maxim\Documents\Studium\Master\Salzburg\Master_Thesis\output\merged_results.csv"
output_folder = r"C:\Users\Maxim\Documents\Studium\Master\Salzburg\Master_Thesis\output"
output_csv = os.path.join(output_folder, 'cluster_assignments_allk.csv')
dendro_output_dir   = os.path.join(output_folder, 'dendogram')
os.makedirs(dendro_output_dir, exist_ok=True)


# 1. Data preparation

# Read and pivot
df_raw = pd.read_csv(input_csv)
thresholds = sorted(df_raw['Threshold'].unique())

def pivot(df, col, prefix):
    p = (df.pivot(index='City', columns='Threshold', values=col)
          .fillna(0)
          .sort_index(axis=1))
    p.columns = [f"{prefix}_{thr}" for thr in p.columns]
    return p

# Create datasets
df_bike   = pivot(df_raw, 'percentage_edges_classes', 'bike')
df_reach  = pivot(df_raw, 'reachable_percentage', 'reach')
df_alpha  = pivot(df_raw, 'Overall Alpha-Index', 'alpha')
df_beta   = pivot(df_raw, 'Overall Beta-Index', 'beta')
df_gamma  = pivot(df_raw, 'Overall Gamma-Index', 'gamma')
df_conn   = pd.concat([df_alpha, df_beta, df_gamma], axis=1)
df_combined = pd.concat([df_bike, df_reach, df_conn], axis=1)

datasets = {
    'bike_index':    df_bike,
    'accessibility': df_reach,
    'connectivity':  df_conn,
    'combined_raw':  df_combined
}

scaler = StandardScaler()
assignments = pd.DataFrame(index=df_bike.index)

# 2. Clustering loop
for name, df_feat in datasets.items():
    X = scaler.fit_transform(df_feat)
    Z = linkage(X, method='ward', metric='euclidean')
    
    for k in range(2, 9):
        labels = fcluster(Z, t=k, criterion='maxclust')
        assignments[f"{name}_k{k}"] = labels

# Save assignments
assignments.reset_index(inplace=True)
assignments.rename(columns={'index': 'City'}, inplace=True)
assignments.to_csv(output_csv, index=False)
print(f"Saved cluster assignments to: {output_csv}")

# 3. Calculate and save scores
score_lines = []
for name, df_feat in datasets.items():
    X = scaler.fit_transform(df_feat)
    Z = linkage(X, method='ward', metric='euclidean')

    for k in range(2, 9):
        labels = fcluster(Z, t=k, criterion='maxclust')
        sil = silhouette_score(X, labels) if len(set(labels)) > 1 else np.nan
        ch  = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else np.nan
        score_lines.append(f"{name} – k={k}: Silhouette = {sil:.3f}, Calinski-Harabasz = {ch:.1f}")

score_path = os.path.join(output_folder, 'clustering_scores.txt')
with open(score_path, 'w') as f:
    f.write("Clustering Evaluation Scores (Silhouette & Calinski-Harabasz)\n")
    f.write("-----------------------------------------------------------\n")
    f.write("\n".join(score_lines))
print(f"Saved clustering evaluation scores to: {score_path}")

# 4. Create and store dendogram
hex_cols = {
    'orange': '#ff7f0e',
    'green':  '#2ca02c',
    'red':    '#d62728',
    'purple': '#9467bd',
    'blue':   '#1f77b4'
}

def plot_dendro_gray(df_feat, title, k, save_path):
    X_scaled = StandardScaler().fit_transform(df_feat)
    Z = linkage(X_scaled, method='ward')

    labels_raw = fcluster(Z, t=k, criterion='maxclust')
    labels     = pd.factorize(labels_raw)[0] + 1
    city2cl    = dict(zip(df_feat.index, labels))

    fig = plt.figure(figsize=(10, 16))
    gs  = fig.add_gridspec(1, 3, width_ratios=[0.68, 0.04, 0.28], wspace=0.05)
    ax_den = fig.add_subplot(gs[0])
    ax_dot = fig.add_subplot(gs[1])
    ax_lab = fig.add_subplot(gs[2])

    # --- Dendrogramm -------------------------------------------------
    dendro = dendrogram(
        Z,
        labels=None,
        orientation='left',
        link_color_func=lambda _: 'grey',
        ax=ax_den
    )
    ax_den.set_xlabel('Euclidean distance')
    ax_den.set_yticks([])
    ax_den.invert_yaxis()
    for sp in ax_den.spines.values():
        sp.set_visible(False)

    ordered = df_feat.index[dendro['leaves']]

    # --- Punkt-Achse -------------------------------------------------
    ax_dot.set_ylim(0, len(ordered))
    ax_dot.invert_yaxis()
    ax_dot.set_xticks([]); ax_dot.set_yticks([])
    for sp in ax_dot.spines.values():
        sp.set_visible(False)

    # --- Label-Achse -------------------------------------------------
    ax_lab.set_ylim(0, len(ordered))
    ax_lab.invert_yaxis()
    ax_lab.set_xticks([]); ax_lab.set_yticks([])
    for sp in ax_lab.spines.values():
        sp.set_visible(False)

    for i, city in enumerate(ordered):
        # Point in grey (not cluster-colored)
        ax_dot.scatter(0.5, i + 0.5, s=18, color='grey')
        ax_lab.text(0, i + 0.5, city, va='center', fontsize=7)

    plt.suptitle(title, y=0.92, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, format='svg')
    plt.close()

dendro_configs = [
    ("Bikeability",   df_bike, 4),
    ("Accessibility", df_reach, 4),
    ("Connectivity",  df_conn, 5),
    ("Combined",      df_combined, 4)
]

for title, df_feat, k in dendro_configs:
    filename = title.lower().replace(" ", "_") + ".svg"
    save_path = os.path.join(dendro_output_dir, filename)
    plot_dendro_gray(df_feat, f"{title} (k={k})", k, save_path)
    print(f"Dendrogram saved: {save_path}")