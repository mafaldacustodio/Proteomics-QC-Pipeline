############################################### IMPORTS ######################################
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
import matplotlib.lines as mlines
from sklearn.decomposition import PCA
from itertools import combinations
from adjustText import adjust_text
import re
import copy
from matplotlib.ticker import FuncFormatter
import scipy.stats as st
from scipy.stats import gaussian_kde

############################################### RENAME PG MATRIX COLUMN NAMES ######################
def rename_mzML_columns(input_file, output_file):
    """
    Reads the protein group matrix, and renames the columns according to the sample ID.
    """
    df = pd.read_csv(input_file, sep="\t")
    # Regex pattern to extract sample ID from filenames 
    pattern = re.compile(r"(?P<id>(S_?\d+|pool\d+_\d+))\.mzML$")
    new_columns = {}
    for col in df.columns:
        if ".mzML" in col:
            basename = col.split("/")[-1]  # Extract file basename
            match = pattern.search(basename)
            if match:
                sample_id = match.group("id")
                sample_id = sample_id.replace("S_", "S") 
                # Ensure sample IDs like 'S1' become 'S001' for consistency
                if sample_id.startswith('S') and sample_id[1:].isdigit():
                    number_part = sample_id[1:]
                    sample_id = f"S{number_part.zfill(3)}"
                new_columns[col] = sample_id 
    df.rename(columns=new_columns, inplace=True)
    # Save the modified DataFrame
    df.to_csv(output_file, sep="\t", index=False)
    return df

############################## NORMALIZATION FUNCTIONS - MEDIAN SCALING ########################

def median_scaling(mat, scale=True):
    mat_median_scaled = pd.DataFrame(
            median_scale(mat.values, scale=scale),
            index=mat.index,
            columns=mat.columns)
    return mat_median_scaled.T

def median_scale(mat, scale): 
    # Step 1: Per-column medians
    col_medians = np.nanmedian(mat, axis=0)
    # Step 2: Global median across all columns
    global_median = np.nanmedian(col_medians)
    # Step 3: Adjustment per column (how much to shift)
    adjval = col_medians - global_median
    #Step 4: Shift every sample's median and
    #media absolute value normalization
    mat_centered = normalize_median_abs_values(mat - adjval)
    return mat_centered

def normalize_median_abs_values(x): 
    # Step 1: Median absolute value for each column
    cmed = np.log(np.nanmedian(np.abs(x), axis=0))
    # Step 2: geometric mean of median absolute values
    cmed = np.exp(cmed - np.mean(cmed))
    # Step 3: Scale each column
    return (x / cmed)


############################## FILTERING FUNCTIONS ########################

def filter_proteins_by_threshold(dataset, threshold=0.20):
    """
    Filters proteins based on the percentage of samples in which they are present.
    """
    print(f"Number of proteins before filtering: {len(dataset.mat.columns)}")
    # Identify valid proteins present in at least 'threshold' proportion of samples
    valid_proteins = dataset.mat.columns[dataset.mat.notnull().mean() >= threshold]
    # Keep only the filtered proteins
    dataset.mat = dataset.mat[valid_proteins]
    print(f"After filtering for {threshold*100:.0f}%, {len(dataset.mat.columns)} proteins remained.")
    return dataset

def select_grps(mat, grps, percent, n=1):
    """
    Filters a matrix based on the presence of values within groups.
    """
    # Split matrix into groups
    grouped = {grp: mat.loc[:, grps == grp] for grp in grps.unique()}
    # Compute presence proportion for each group
    presence_matrix = pd.DataFrame({grp: (df.notna().sum(axis=1) / df.shape[1]) >= percent
                                    for grp, df in grouped.items()})
    # Keep rows where at least 'n' groups meet the threshold
    mat_filtered = mat.loc[presence_matrix.sum(axis=1) >= n, :]

    return mat_filtered

############################## FUNCTIONS ##############################

def generate_barplot_thresholds(ds, output_dir):
    """
    Generate a barplot of protein counts across different filtering thresholds.
    """
    thresholds = [0.0, 0.2, 0.5, 0.7, 1.0]
    protein_counts = []
    ds_orig = copy.deepcopy(ds)
    for thresh in thresholds:
        ds_filtered = filter_proteins_by_threshold(ds_orig, thresh)
        protein_counts.append(ds_filtered.mat.shape[1])
    plt.figure(figsize=(8, 5))
    sns.barplot(x=[f"{int(t*100)}%" for t in thresholds], y=protein_counts, palette="Blues_d")
    plt.xlabel("Filtering Threshold")
    plt.ylabel("Number of Proteins")
    plt.title("Protein Count Across Filtering Thresholds")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    output_path = os.path.join(output_dir, "protein_filtering_by_threshold.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

def generate_sample_distribution_plot(ds, output_dir):
    """
    Generate a sample distribution plot, save it, and return outlier distribution string.
    """
    sample_totals = ds.mat.sum(axis=1)
    mean_total = sample_totals.mean()
    std_total = sample_totals.std()
    z = st.norm.ppf(0.99)
    lower = mean_total - z * std_total
    upper = mean_total + z * std_total
    outliers = sample_totals[(sample_totals < lower) | (sample_totals > upper)]
    outliers_distribution = ','.join(outliers.index.astype(str))
    fig = ds.plot_sampledistribution(method='box', color='plate')
    x_labels = ds.mat.index.tolist()
    x_colors = ['red' if sample in outliers else 'black' for sample in x_labels]
    fig.update_layout(width=1800, height=800, showlegend=False, margin=dict(l=50, r=50, t=50, b=150))
    fig.update_xaxes(tickangle=90, tickmode='array', tickvals=x_labels,
                     ticktext=[f"<span style='color:{c}'>{label}</span>" for label, c in zip(x_labels, x_colors)])
    fig.write_image(f"{output_dir}/sample_distribution_outliers.png")
    plt.close()
    return outliers_distribution

def plot_missing_values_heatmap(data, output_dir, filename="missing_data_heatmap.png", dpi=300, figsize=(15, 12), cmap='Blues'):
    """
    Plots and saves a heatmap of missing values in the data.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(data.isnull().transpose(), cmap=cmap, cbar=False)
    plt.title("Missing Values")
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi)
    plt.close()

def plot_missing_values_per_group(ds, group_col, output_dir, filename="missing_values_per_group.png", figsize=(10, 6), dpi=300):
    """
    Plots a bar plot of the percentage of missing proteins per sample grouped by a specified condition.
    """
    # Compute the percentage of missing values for each sample (row in ds.mat)
    missing_pct = ds.mat.isnull().mean(axis=1) * 100
    df_missing = pd.DataFrame({"missing_pct": missing_pct})
    
    # Merge with metadata using "analytical_id" for merging
    metadata = ds.metadata.copy()
    df_missing["analytical_id"] = df_missing.index
    df_missing = pd.merge(df_missing, metadata[["analytical_id", group_col]], on="analytical_id", how="left")
    
    # Exclude rows with missing group values
    df_missing = df_missing.dropna(subset=[group_col])
    
    # Get unique conditions in sorted order
    unique_conditions = sorted(df_missing[group_col].unique())
    
    # Create a figure with one subplot per condition, stacked vertically.
    n_conditions = len(unique_conditions)
    fig_height = figsize[1] * n_conditions
    fig, axes = plt.subplots(nrows=n_conditions, ncols=1, figsize=(figsize[0], fig_height), sharex=False)
    
    # If only one condition, ensure axes is iterable:
    if n_conditions == 1:
        axes = [axes]
    
    # Loop through each condition and plot the barplot for that condition
    for ax, cond in zip(axes, unique_conditions):
        sub_df = df_missing[df_missing[group_col] == cond]
        # Bar plot: x-axis is analytical_id and y-axis is missing percentage
        ax.bar(sub_df["analytical_id"], sub_df["missing_pct"], color=sns.color_palette("Set2", n_conditions)[unique_conditions.index(cond)])
        ax.set_title(f"{group_col}: {cond}", fontsize=12)
        ax.set_ylabel("Missing (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=90, labelsize=8)
        # Optionally, you could annotate the bars if needed
        
    # Set a common x-label across the figure; since each subplot is different, using a common xlabel
    plt.xlabel("Sample")
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()
  


def plot_missing_proteins_distribution_by_group(ds, group_col, output_dir, filename="missing_proteins_distribution.png",  dpi=300):
    """
    Computes the number of missing proteins per sample from ds.mat, joins this info with the metadata
    based on the 'analytical_id' column, and plots a faceted histogram (with KDE) of missing protein counts 
    for each biological group specified by group_col.
    """
    # Compute the number of missing proteins per sample
    missing_counts = ds.mat.isnull().sum(axis=1)
    df_missing = pd.DataFrame({"missing_counts": missing_counts})
    # Merge with metadata using "analytical_id"
    metadata = ds.metadata.copy().set_index("analytical_id")
    df_missing = df_missing.join(metadata[[group_col]])
    # Compute global x-axis limits using missing_counts values
    global_xmin = df_missing["missing_counts"].min()
    global_xmax = df_missing["missing_counts"].max()
    # Create FacetGrid with shared x-axis and y-axis scales across facets
    g = sns.FacetGrid(df_missing, col=group_col, col_wrap=4, sharex=True, sharey=True, height=4)
    g.map(sns.histplot, "missing_counts", kde=True, bins=20, color="steelblue")
    # Set same x-axis limits for all facets
    g.set(xlim=(global_xmin, global_xmax))
    g.fig.suptitle("Distribution of Missing Protein Counts per Group", x=0.0, ha="left")
    g.set_axis_labels("Missing Protein Count", "Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi)
    plt.show()


def plot_sample_correlation_heatmap(data, output_dir, filename="sample_correlation_heatmap.png", dpi=300, figsize=(12, 12), cmap="vlag", method='spearman'):
    """
    Plots a heatmap of the sample correlation.
    """
    correlation_matrix = data.corr(method=method)
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, cmap=cmap)
    plt.title("Correlation Matrix")
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi)
    plt.close()


def plot_hierarchical_clustering(data, output_dir, sample_annotation=None, factors_to_plot=None, distance_metric="euclidean", method="ward", 
                                 filename="sample_dendrogram.png", width=16, height=8):
    """
    Performs hierarchical clustering on data and plots a dendrogram with one colored strip per factor
    """
    # Compute linkage matrix
    linkage_data = linkage(data, method=method, metric=distance_metric)

    #number of factors
    n_factors = len(factors_to_plot) if factors_to_plot is not None else 0
    total_rows = 1 + n_factors  # One row for dendrogram, one per factor

    # Create figure
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(total_rows, 1, hspace=0.05)

    # Plot dendrogram in the first row
    ax0 = fig.add_subplot(gs[0, 0])
    dendro = dendrogram(linkage_data, ax=ax0)
    ax0.set_ylabel("Distance")
    ax0.set_xticks([])
    ax0.set_title("Hierarchical Clustering Dendrogram")

    print("PCA index:", data.index.tolist())
    print("Annotation analytical_id:", sample_annotation["analytical_id"].tolist())
    # Get sample order from dendrogram leaves
    sample_order = [data.index[i] for i in dendro['leaves']]

    # If annotations and factors are provided, add one color strip per factor
    if sample_annotation is not None and factors_to_plot is not None:
        for idx, factor in enumerate(factors_to_plot):
            ax = fig.add_subplot(gs[idx+1, 0])
            # Get the factor values in dendrogram order
            factor_values = sample_annotation.set_index('analytical_id').loc[sample_order, factor]

            # Generate random colors for each unique value in this factor
            unique_values = factor_values.unique()
            factor_color_map = {val: np.random.rand(3) for val in unique_values}
            # Map each sample's factor value to a color
            strip_colors = factor_values.map(factor_color_map)
            # Convert to a 1xN array (each row is an RGB triplet)
            color_array = np.array(strip_colors.tolist()).reshape(1, -1, 3)
            ax.imshow(color_array, aspect='auto')
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim(0, len(strip_colors))
            ax.set_title(factor, fontsize=10)
            # Add legend for this factor
            handles = [mlines.Line2D([], [], marker='s', color=factor_color_map[val],linestyle='', markersize=10, label=str(val))for val in unique_values]
            ax.legend(handles=handles, title=factor, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.show()
    return fig


def plot_all_pcas_by_factor(data, sample_annotation, factors_to_plot, n_clusters, output_dir, filename="pcas_by_factor.png", dpi=300):
    """
    Computes a 2-component PCA on the data and creates one scatter plot per factor.
    In each subplot, points are colored by clusters (obtained from hierarchical clustering)
    and shaped by the factor's value. All PCA plots are saved in a single image.
    """
    # Compute linkage matrix using hierarchical clustering
    linkage_data = linkage(data, method="ward",metric="euclidean")
    # Generate cluster labels using fcluster
    clusters = fcluster(linkage_data, t=n_clusters, criterion="maxclust")
    
    # Compute PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data.values)
    pca_df = pd.DataFrame(pca_result, index=data.index, columns=["PC1", "PC2"])
    pca_df["cluster"] = clusters

    # Determine plot dimensions based on the number of factors
    n_factors = len(factors_to_plot)
    n_cols = math.ceil(math.sqrt(n_factors))
    n_rows = math.ceil(n_factors / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows))
    if n_factors == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    # Define a color palette for clusters
    unique_clusters = np.unique(clusters)
    cluster_colors = dict(zip(unique_clusters, sns.color_palette("husl", len(unique_clusters))))
    
    # Marker choices for factors (if more markers are needed, add to the list)
    marker_choices = ['o', 's', 'D', '^', 'v', '<', '>', 'P', 'X']

    # Create one subplot per factor
    for i, factor in enumerate(factors_to_plot):
        ax = axes[i]
        # Add the factor values to the PCA DataFrame
        print(pca_df)
        pca_df[factor] = sample_annotation.set_index('analytical_id').loc[pca_df.index, factor]
        unique_factor_vals = pca_df[factor].unique()
        print(pca_df)
        # Assign a marker for each unique factor value
        marker_map = {val: marker_choices[j % len(marker_choices)] for j, val in enumerate(unique_factor_vals)}
        
        # Plot each sample individually with color from cluster and marker from factor value
        for idx, row in pca_df.iterrows():
            clust = row["cluster"]
            marker = marker_map[row[factor]]
            ax.scatter(row["PC1"], row["PC2"], color=cluster_colors[clust],
                       marker=marker, edgecolor='k', s=50)
        
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"PCA (Shape by {factor})")
        
        # Create legend entries: one for cluster colors and one for marker shapes
        cluster_handles = [mlines.Line2D([], [], color=cluster_colors[c], marker='o', linestyle='',
                                         markersize=10, label=f"Cluster {c}") for c in unique_clusters]
        factor_handles = [mlines.Line2D([], [], color='gray', marker=marker_map[val], linestyle='',
                                        markersize=10, label=f"{factor}={val}") for val in unique_factor_vals]
        ax.legend(handles=cluster_handles + factor_handles, fontsize=8)
    
    # Remove any extra subplots if grid is larger than needed
    for j in range(n_factors, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=dpi, bbox_inches='tight')
    plt.show()
    return fig


def plot_plate_distances(data, metadata, output_dir, dpi=300):
    """
    Computes Euclidean distances within and between plates and creates a boxplot.
    """
    # Extract plate information and compute distance matrix
    plates = metadata['plate'].values
    dmat = squareform(pdist(data, metric='euclidean'))
    unique_plates = np.unique(plates)
    
    within = []
    between = []
    for p in unique_plates:
        idx = np.where(plates == p)[0]
        if len(idx) >= 2:
            within.extend(pdist(data.iloc[idx], metric='euclidean'))
    for p1, p2 in combinations(unique_plates, 2):
        idx1 = np.where(plates == p1)[0]
        idx2 = np.where(plates == p2)[0]
        for i in idx1:
            for j in idx2:
                between.append(dmat[i, j])
    df = pd.DataFrame({ "Distance Type": ["Within-plate"] * len(within) + ["Between-plate"] * len(between),"Distance": within + between})
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Distance Type", y="Distance", palette="Set2")
    plt.title("Within vs Between Plate Distances")
    plt.ylabel("Euclidean Distance")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "distances_batches.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return out_path

def plot_enrichment_pcs(data,output_dir, top_n=10, protein_map=None, filename="pc1_pc2_enrichment.png"):
    """
    Performs PCA on the given data and plots the top contributing features for PC1 and PC2.
    """
    # Perform PCA on the data (using all features)
    pca = PCA(n_components=2)
    pca.fit(data)
    # The loadings (components) have shape: (n_components, n_features)
    # Create a DataFrame of loadings using the features as the index.
    loadings = pd.DataFrame(pca.components_.T, index=data.columns, columns=["PC1", "PC2"])
    
    # 4. Replace raw IDs with more readable names
    if protein_map is not None:
        loadings.index = [protein_map.get(feat, feat) for feat in loadings.index]
    
    # Prepare the figure: one subplot for PC1 and one for PC2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, pc in enumerate(["PC1", "PC2"]):
        ax = axes[i]
        # Select top_n features based on absolute loading values for this PC
        top_features = loadings[pc].abs().sort_values(ascending=False).head(top_n)
        # Keep the actual loading values (with sign) but order them by absolute value (largest first)
        ordered_features = top_features.index.tolist()
        df_top = loadings.loc[ordered_features, pc]
        
        # Create an array of positions for plotting (one per feature)
        y_positions = np.arange(len(df_top))
        
        # Scale dot sizes based on the absolute loading values (relative to the maximum)
        sizes = (df_top.abs() / df_top.abs().max()) * 500  # Adjust scale factor as needed
        
        # Set colors: red if loading is positive, blue if negative
        colors = ["red" if x > 0 else "blue" for x in df_top.values]
        
        # Plot dots: x = loading, y = corresponding position
        ax.scatter(df_top.values, y_positions, s=sizes, c=colors, alpha=0.8, edgecolor="k")
        # Set y-ticks as feature names
        ax.set_yticks(y_positions)
        ax.set_yticklabels(ordered_features)
        # Invert the y-axis so that the top contributor (highest absolute value) appears at the top
        ax.invert_yaxis()
        ax.set_xlabel("Loading Value")
        ax.set_title(f"Top {top_n} Contributors to {pc}", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    




def compute_outliers(protein_data, z_value=st.norm.ppf(0.99)):
    """
    Computes outliers for each sample (column) in protein_data based on:
      - x-axis: sample means (99% CI)
      - y-axis: mean density from KDE (99% CI)
    Returns a list of samples that are outliers by either criterion.
    """
    # Outliers based on x-axis (sample means)
    sample_means = protein_data.mean(axis=0)
    overall_mean_x = sample_means.mean()
    overall_std_x = sample_means.std()
    lower_x = overall_mean_x - z_value * overall_std_x
    upper_x = overall_mean_x + z_value * overall_std_x
    outliers_x = sample_means[(sample_means < lower_x) | (sample_means > upper_x)].index.tolist()

    # Outliers based on y-axis (mean density from KDE)
    x_min = protein_data.min().min()
    x_max = protein_data.max().max()
    grid = np.linspace(x_min, x_max, 200)
    density_means = {}
    for sample in protein_data.columns:
        values = protein_data[sample].dropna().values
        if len(values) < 2:
            density_means[sample] = np.nan
        else:
            kde = gaussian_kde(values)
            density_means[sample] = kde(grid).mean()
    density_series = pd.Series(density_means)
    overall_mean_y = density_series.mean()
    overall_std_y = density_series.std()
    lower_y = overall_mean_y - z_value * overall_std_y
    upper_y = overall_mean_y + z_value * overall_std_y
    outliers_y = density_series[(density_series < lower_y) | (density_series > upper_y)].index.tolist()

    # Combine outliers from both criteria (union)
    return list(set(outliers_x) | set(outliers_y))

def plot_density_per_sample(ds, output_dir, dpi=300):
    """
    Plots density curves for each sample in ds.mat and highlights outlier samples.
    """
    protein_data = ds.mat.T
    outliers_combined = compute_outliers(protein_data)
    
    plt.figure(figsize=(12, 6))
    handles, labels = [], []
    for sample in protein_data.columns:
        ax = sns.kdeplot(protein_data[sample].dropna(), alpha=0.6)
        line_color = ax.lines[-1].get_color()
        if sample in outliers_combined:
            handles.append(mlines.Line2D([], [], color=line_color, label=sample))
            labels.append(sample)
    plt.xlabel("Log2 Protein Intensity")
    plt.ylabel("Density")
    plt.title("Log2 Protein Intensity Distribution per Sample")
    if handles:
        plt.legend(handles=handles, labels=labels, loc='center left',bbox_to_anchor=(1, 0.5), title="Potential Outliers")
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, "protein_density_samples.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    if outliers_combined:
        print(', '.join(outliers_combined))
    else:
        print("No samples deviate significantly.")
    return outliers_combined

def plot_mean_intensity_density(ds, output_dir, outliers_combined=None, dpi=300):
    """
    Plots the mean protein intensity per sample and overlays density curves for outlier samples.
    """
    protein_data = ds.mat.T
    mean_intensity = protein_data.mean(axis=1)
    if outliers_combined is None:
        outliers_combined = compute_outliers(protein_data)
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(mean_intensity, color="black", linewidth=2, label="Mean Intensity")
    handles, labels = [], []
    for sample in outliers_combined:
        if sample in protein_data.columns:
            ax = sns.kdeplot(protein_data[sample].dropna(), alpha=0.6)
            line_color = ax.lines[-1].get_color()
            handles.append(mlines.Line2D([], [], color=line_color, label=sample))
            labels.append(sample)
    if handles:
        plt.legend(handles=handles, labels=labels, loc='center left',bbox_to_anchor=(1, 0.5), title="Potential Outliers")
    plt.xlabel("Log2 Mean Protein Intensity with Outliers")
    plt.ylabel("Density")
    plt.title("Log2 Protein Intensity Distribution")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mean_protein_density.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_intensity_histogram_with_imputed(ds, ds_imputed, output_dir, dpi=300):
    """
    Plots histograms of mean protein intensity for non-imputed and imputed data.
    """
    protein_data = ds.mat.T
    mean_non_imputed = protein_data.mean(axis=1)
    mean_imputed = ds_imputed.mat.T.mean(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(mean_imputed, bins=30, color='orange', label="Imputed", edgecolor='black')
    plt.hist(mean_non_imputed, bins=30, color='blue', label="Non-Imputed", edgecolor='black')
    plt.xlabel("Log2 Mean Protein Intensity")
    plt.ylabel("Frequency")
    plt.title("Mean Log2 Protein Intensity Distribution")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(output_dir, "mean_protein_histogram.png")
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def plot_albumin_concentration(ds, output_dir, albumin_id="P02768", dpi=300):
    """
    Creates a scatterplot of albumin intensity across samples using the specified albumin_id
    from ds.mat, identifies outliers based on a 99% confidence interval, annotates them,
    and saves the plot.
    """
    # Retrieve albumin intensities and sample labels
    albumin_values = ds.mat[albumin_id]
    sample_labels = ds.mat.index

    # Calculate 99% confidence interval using z-score from norm.ppf(0.997)
    z = st.norm.ppf(0.99)
    mean_alb = albumin_values.mean()
    std_alb = albumin_values.std()
    lower_alb = mean_alb - z * std_alb
    upper_alb = mean_alb + z * std_alb

    # Identify outlier samples
    outlier_mask = (albumin_values < lower_alb) | (albumin_values > upper_alb)
    outliers = albumin_values[outlier_mask]
    albumin_outliers_str = ", ".join(outliers.index.astype(str))
    positions = np.arange(len(sample_labels))

    # Create scatterplot: blue for normal, red for outliers
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=positions, y=albumin_values, color='blue', s=50, alpha=0.7, label="Normal")
    outlier_positions = [positions[sample_labels.get_loc(s)] for s in outliers.index]
    sns.scatterplot(x=outlier_positions, y=outliers.values, color='red', s=50, alpha=0.7, label="Outlier")
    
    # Annotate outlier sample names
    for sample in outliers.index:
        pos = sample_labels.get_loc(sample)
        plt.text(pos, albumin_values.loc[sample], sample, color='red', fontsize=8, va='bottom', ha='right')
    
    plt.xlabel("Samples")
    plt.ylabel("Albumin Intensity")
    plt.xticks([])  
    plt.title("Albumin Intensity Across Samples")
    plt.legend()
    plt.tight_layout()
 
    out_path = os.path.join(output_dir, "albumin_scatterplot.png")
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    
    return out_path, albumin_outliers_str


def plot_protein_rank_abundance(ds, output_dir, filename="protein_rank_abundance_plot.png", figsize=(14, 6)):
    """
    Generates a protein rank-abundance plot by mapping protein group IDs to names,
    computing mean abundances, and annotating the top and bottom 10 proteins.
    """
    # Map Protein.Group to Protein.Names from raw input
    protein_group_to_name = dict(zip(ds.rawinput["Protein.Group"], ds.rawinput["Protein.Names"]))
    
    # Compute mean abundance across samples and sort in descending order
    mean_protein_abundance = ds.mat.mean(axis=0).sort_values(ascending=False)
    
    # Replace protein group IDs with readable names
    named_proteins = mean_protein_abundance.copy()
    named_proteins.index = [protein_group_to_name.get(pg, pg) for pg in mean_protein_abundance.index]
    
    # Plot scatter plot of protein rank vs. mean abundance
    plt.figure(figsize=figsize)
    sns.scatterplot(x=range(1, len(named_proteins) + 1), y=named_proteins.values,color='steelblue', s=40)
    
    texts = []
    # Annotate top 10 proteins
    for i in range(10):
        name = named_proteins.index[i]
        y = named_proteins.iloc[i]
        texts.append(plt.text(i + 1, y, name, fontsize=9, ha='left', va='bottom', color='darkred'))
    # Annotate bottom 10 proteins
    for i in range(1, 11):
        idx = len(named_proteins) - i
        name = named_proteins.index[idx]
        y = named_proteins.iloc[idx]
        texts.append(plt.text(idx + 1, y, name, fontsize=9, ha='left', va='top', color='darkgreen'))
    
    # Adjust annotations to reduce overlap
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color='gray', lw=0.5))
    
    plt.xlabel("Protein Rank")
    plt.ylabel("Mean Log2 Intensity")
    plt.title("Protein Rank-Abundance Plot")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    rank_plot_path = os.path.join(output_dir, filename)
    plt.savefig(rank_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return rank_plot_path


#Compute Protein Rank Order
def compute_protein_rank_order(ds, protein_map=None):
    """
    Computes the mean abundance per protein and returns the ordered list of protein names.
    Optionally uses a mapping (protein_map) to convert raw IDs to readable protein names.
    """
    mean_protein_abundance = ds.mat.mean(axis=0).sort_values(ascending=False)
    protein_order = mean_protein_abundance.index.tolist()
    return protein_order

#Compute Intra-individual CV 
def compute_intra_cv(ds,  group_filter=None):
    """
    Computes the intra-individual CV for each protein (using repeated samples per individual).
    For each protein, groups samples by 'biological_id' (assumed to be in ds.metadata) and computes
    CV = (std/mean)*100 for individuals with replicates, then averages these CVs across individuals.
   
    """
    df = ds.mat.copy()
    print("DS INTRA CV BEFORE CHANGING INDEX", ds)
    df["analytical_id"] = df.index
    print("DS INTRA CV AFTER CHANGING INDEX", ds)
    meta = ds.metadata.copy()
    print("METADATA INTRA CV BEFORE CHANGES", meta)
    df_meta = pd.merge(df, meta, on="analytical_id", how="left")
    print("METADATA INTRA CV after CHANGES", meta)

    if group_filter is not None:
        for key, value in group_filter.items():
            df_meta = df_meta[df_meta[key] == value]
    
    def cv_for_individual(x):
         # Check if there is repeated samples/values and the mean is not zero.
        if len(x) > 1 and np.mean(x) != 0:
            return np.std(x, ddof=1) / np.mean(x) * 100
        else:
            return np.nan

    intra_cv = {}
    for protein in ds.mat.columns:
        # Group the merged data by "biological_id" (each individual).
        # Then, select the column corresponding to the current protein.
        grouped = df_meta.groupby("biological_id")[protein]
        # Apply the cv_for_individual function to each group, calculating a CV per individual.
        cvs = grouped.apply(cv_for_individual)
        intra_cv[protein] = cvs.mean()  # mean across individuals, ignoring NaNs
    return pd.Series(intra_cv)

#Compute Inter-individual CV 
def compute_inter_cv(ds, group_filter=None):
    """
    Computes the inter-individual CV for each protein.
    For each protein, first computes the mean intensity per individual (grouping by biological_id),
    then calculates CV = (std/mean)*100 across individuals.
    """
    df = ds.mat.copy()
    df["analytical_id"] = df.index
    meta = ds.metadata.copy()
    df_meta = pd.merge(df, meta, on="analytical_id", how="left")

    if group_filter is not None:
        for key, value in group_filter.items():
            df_meta = df_meta[df_meta[key] == value]
    
    inter_cv = {}
    for protein in ds.mat.columns:
        # Group the data by "biological_id" and compute the mean intensity for each individual.
        indiv_means = df_meta.groupby("biological_id")[protein].mean()
        if indiv_means.mean() != 0:
            #computes the cv across all individuals 
            inter_cv[protein] = np.std(indiv_means, ddof=1) / indiv_means.mean() * 100
        else:
            inter_cv[protein] = np.nan
    return pd.Series(inter_cv)

#Plot Scatter for CV vs Protein Rank 
def plot_cv_scatter(cv_series, protein_order, title, output_file, figsize=(12,6)):
    """
    Plots a scatter plot of a CV Series (with proteins as index) against protein rank.
    The protein order is enforced using the provided protein_order list.
    """
    cv_series = cv_series.reindex(protein_order)
    
    plt.figure(figsize=figsize)
    plt.scatter(range(1, len(cv_series)+1), cv_series.values, color="lightblue", edgecolor="navy", alpha=0.8)
    plt.xlabel("Protein Rank")
    plt.ylabel("CV (%)")
    plt.title(title)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()
    return output_file

#Plot CV Scatter Plots by Condition in a Single Figure
def plot_cv_scatter_by_conditions(ds, group_col, protein_order, cv_func, output_file, base_figsize=(12,6), dpi=300):
    """
    For each unique condition specified in ds.metadata[group_col], computes the CV (using cv_func)
    and plots a scatter plot (CV vs protein rank) in a vertically stacked subplot.
    All subplots share the same protein order (from protein_order) and are saved in one single figure.
    """
    conditions = sorted(ds.metadata[group_col].dropna().unique())
    n_conditions = len(conditions)
    
    fig_width = base_figsize[0]
    fig_height = base_figsize[1] * n_conditions
    fig, axes = plt.subplots(nrows=n_conditions, ncols=1, figsize=(fig_width, fig_height), sharex=True)
    if n_conditions == 1:
        axes = [axes]
    
    for ax, cond in zip(axes, conditions):
        cv_series = cv_func(ds, group_filter={group_col: cond})
        cv_series = cv_series.reindex(protein_order)
        x = np.arange(1, len(cv_series)+1)
        y = cv_series.values
        ax.scatter(x, y, color="lightblue", edgecolor="navy", alpha=0.8)
        ax.set_ylabel("CV (%)")
        ax.set_title(f"{cv_func.__name__.replace('compute_','').replace('_cv','').upper()} CV ({cond})", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
    
    plt.xlabel("Protein Rank")
    plt.xticks([])
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=dpi, bbox_inches="tight")
    plt.show()
    plt.close()
    
    return output_file

################################ Contamination Panel ###########################################

def prepare_data_contamination(pg_matrix_new, contamination_panel):
    combined_data = pd.read_csv(pg_matrix_new, sep='\t')
    panel = pd.read_excel(contamination_panel)
    combined_data['genes'] = combined_data['Genes'].str.upper()
    panel['genes'] = panel['genes'].str.upper()
    non_sample_cols = ['Protein.Group', 'Protein.Names', 'First.Protein.Description', 'Genes', 'genes']
    sample_cols = [col for col in combined_data.columns if col not in non_sample_cols]
    merged_data = (combined_data.melt(id_vars=['genes'], value_vars=sample_cols, var_name='Sample', value_name='RawIntensity')
                   .merge(panel[panel['best_markers'] == 1], on='genes'))
    quality_marker_order = ["coagulation", "erythrocyte", "platelet"]
    gene_order = ["FGA", "FGB", "FGG", "HBD", "HBB", "CA1", "CA2", "CAT", "BLVRB", "PRDX2", "TPM4", "TLN1", "MYH9"]
    merged_data['quality_marker'] = pd.Categorical(merged_data['quality_marker'], categories=quality_marker_order, ordered=True)
    merged_data['genes'] = pd.Categorical(merged_data['genes'], categories=gene_order, ordered=True)
    merged_data['Sample'] = pd.Categorical(merged_data['Sample'], categories=merged_data['Sample'].unique(), ordered=True)
    return merged_data

def scientific_notation(x, pos):
    return f'{x:.1e}'

def apply_outlier_thresholds(data, z_value):
    """
    For each gene, compute threshold where z_value corresponds to a 99% confidence interval.
    Flag rows where RawIntensity is above this threshold.
    """
    stats_grp = data.groupby("genes")["RawIntensity"].agg(['mean', 'std']).reset_index()
    stats_grp['threshold'] = stats_grp['mean'] + z_value * stats_grp['std']
    return data.merge(stats_grp[['genes', 'threshold']], on="genes").assign(is_outlier=lambda df: df['RawIntensity'] > df['threshold'])

def plot_contamination_panel(merged_data, output_dir, z_value=3, highlight_outliers=False):
    """
    Plot the contamination panel for each quality_marker.
    If highlight_outliers is True, use a 99% confidence interval (threshold = mean + z_value*std)
    to flag outliers and overlay red "X" markers on those points.
    Returns a string listing the outlier samples if highlighting is enabled.
    """
    sns.set_theme(style="whitegrid")
    if highlight_outliers:
        merged_data = apply_outlier_thresholds(merged_data, z_value)
    
    g = sns.FacetGrid(merged_data, col="quality_marker", col_wrap=1, sharey=False)
    g.map_dataframe(sns.lineplot, x="Sample", y="RawIntensity", hue="genes", linewidth=1)
    g.map_dataframe(sns.scatterplot, x="Sample", y="RawIntensity", hue="genes", s=100, alpha=0.7)
    
    if highlight_outliers:
        for ax, (marker, sub_data) in zip(g.axes.flat, merged_data.groupby("quality_marker")):
            sub_outliers = sub_data[sub_data["is_outlier"]]
            if not sub_outliers.empty:
                sns.scatterplot(x="Sample", y="RawIntensity", s=200, marker="X", edgecolor="black",
                                color='red', ax=ax, data=sub_outliers, legend=False)
    
    g.set_axis_labels("Samples", "Raw Intensity").set_titles("{col_name}", size=18)
    g.set(xlim=(-1, len(merged_data['Sample'].unique()))).add_legend()
    for i, ax in enumerate(g.axes.flat):
        ax.yaxis.set_major_formatter(FuncFormatter(scientific_notation))
        ax.tick_params(axis='x', labelrotation=90 if i == len(g.axes.flat) - 1 else 0)
    
    suffix = "_outliers" if highlight_outliers else ""
    plt.suptitle(f"Blood Contamination Panel{suffix}", fontsize=20, x=0.07, y=0.95)
    g.fig.set_size_inches(60, 20)
    plt.subplots_adjust(left=0.05, right=0.97, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(output_dir, f"Blood_Contamination_Panel{suffix}.png"), dpi=300)
    plt.close()
    
    if highlight_outliers:
        outlier_samples = merged_data.loc[merged_data["is_outlier"], "Sample"].unique()
        outlier_string = ", ".join(map(str, outlier_samples))
        return outlier_string


################################ PDF CREATIO FUNCTIONS ###########################################

def to_list(var):
    """Convert a comma‐separated string to a list, or return an empty list if invalid."""
    return [s.strip() for s in var.split(",")] if isinstance(var, str) and var.strip() else []

def generate_outlier_table(sd_outliers, density_outliers, pca_nonimputed_euc, pca_nonimputed_mah, pca_imputed_euc, pca_imputed_mah, albumin_outliers, contamination_outliers):
    """Combine outlier lists and return a DataFrame summarizing which samples are flagged for each metric."""
    all_samples = sorted(set(sd_outliers) | set(density_outliers) | set(pca_nonimputed_euc) | 
                         set(pca_nonimputed_mah) | set(pca_imputed_euc) | set(pca_imputed_mah) | 
                         set(albumin_outliers) | set(contamination_outliers))
    table_data = {
        "Sample Distribution": ["x" if s in sd_outliers else "" for s in all_samples],
        "Density": ["x" if s in density_outliers else "" for s in all_samples],
        "PCA ED": ["x" if s in pca_nonimputed_euc else "" for s in all_samples],
        "PCA MD": ["x" if s in pca_nonimputed_mah else "" for s in all_samples],
        "PCA I ED": ["x" if s in pca_imputed_euc else "" for s in all_samples],
        "PCA I MD": ["x" if s in pca_imputed_mah else "" for s in all_samples],
        "Albumin": ["x" if s in albumin_outliers else "" for s in all_samples],
        "Contamination": ["x" if s in contamination_outliers else "" for s in all_samples]
    }
    return pd.DataFrame(table_data, index=all_samples)

def add_image_and_text_page(pdf, image_path, title="", description="", figsize=(10,12)):
    """
    Adds a page to the PDF containing a title and a description (in the top part)
    and the specified image (in the bottom part). The figure is divided vertically into two rows.
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1, 10], hspace=0.1)
    
    # Top subplot for title/description
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis('off')
    # Place the title near the top-left
    ax_text.text(0.0, 0.9, title, ha='left', va='top', fontsize=16, fontweight='bold')
    # Place the description a bit further down
    ax_text.text(0.0, 0.55, description, ha='left', va='top', fontsize=12)
    
    # Bottom subplot for the image
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.axis('off')
    img = plt.imread(image_path)
    ax_img.imshow(img)
    pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close()

def generate_pdf_report(report, output_dir, output_dir_imputed, workflow_cohort, num_proteins_before, num_proteins_after, 
                        outliers_distribution, pca_euclidean_samples, pca_mahanobis_samples,pca_imputed_euclidean_samples, pca_imputed_mahalanobis_samples,
                        albumin_outliers_str,create_contamination, outlier_info, density_outliers, rank_plot_path):
    # Convert outlier strings to lists
    sd_outliers = to_list(outliers_distribution)
    pca_nonimputed_euc = to_list(pca_euclidean_samples)
    pca_nonimputed_mah = to_list(pca_mahanobis_samples)
    pca_imputed_euc = to_list(pca_imputed_euclidean_samples)
    pca_imputed_mah = to_list(pca_imputed_mahalanobis_samples)
    albumin_outliers = to_list(albumin_outliers_str)
    contamination_outliers = to_list(outlier_info)
    
    # Combine all outlier samples (union)
    all_outlier_samples = sorted(set(sd_outliers) | set(density_outliers) | set(pca_nonimputed_euc) | 
                                   set(pca_nonimputed_mah) | set(pca_imputed_euc) | set(pca_imputed_mah) | 
                                   set(albumin_outliers) | set(contamination_outliers))
    # Create outlier summary table
    table_df = generate_outlier_table(sd_outliers, density_outliers, pca_nonimputed_euc, pca_nonimputed_mah, 
                                      pca_imputed_euc, pca_imputed_mah, albumin_outliers, contamination_outliers)
    

    with PdfPages(report) as pdf:
        # Page 1: Title Page
        fig = plt.figure(figsize=(8.27, 11.69))
        plt.axis('off')
        plt.text(0.5, 0.5, f"QC Report for Proteomics Data Analysis\n\n{workflow_cohort}", ha='center', va='center', fontsize=24)
        pdf.savefig(fig)
        plt.close()

        # Page 3: Sample Distribution Plot, with description.
        add_image_and_text_page(pdf,os.path.join(output_dir, "sample_distribution_outliers.png"),title="Sample Distribution Plot",
            description="Distribution of sample intensities after Median Scaling normalization, Log 2 transformation and filtering.",figsize=(24,14))

        # Page 4: Filtering Comparison Plot
        add_image_and_text_page(pdf,os.path.join(output_dir, "protein_filtering_by_threshold.png"),title="Filtering Comparison",
            description="Number of proteins detected across different filtering thresholds.",figsize=(10,10))

        # Page 5: Missing Values Heatmap
        add_image_and_text_page(pdf,os.path.join(output_dir, "missing_data_heatmap.png"),title="Missing Values Heatmap",
            description="Missing values across the dataset.",figsize=(12,12))

        # Page 6: Missing Values per Group
        add_image_and_text_page(pdf,os.path.join(output_dir, "missing_values_per_group.png"),title="Missing Values per Group",
            description="Percentage of missing values per sample, grouped by biological condition.",figsize=(12,14))

        # Page 7: Missing Proteins per Group
        add_image_and_text_page(pdf,os.path.join(output_dir, "missing_proteins_distribution.png"),title="Missing Proteins per Group",
            description="Distribution of missing protein counts for each biological condition.",figsize=(14,8))

        # Page 8: Sample Correlation Heatmap
        add_image_and_text_page(pdf,os.path.join(output_dir, "sample_correlation_heatmap.png"),title="Sample Correlation Heatmap",
            description="Correlation between samples.",figsize=(10,12))

        # Page 9: Sample Dendrogram Plate Strip
        add_image_and_text_page(pdf, os.path.join(output_dir, "sample_dendrogram.png"),title="Hierarchical Clustering Dendrogram",
            description="Hierarchical clustering of samples with plate annotations.",figsize=(14,10))

        # Page 10: Distances Batch Plot
        add_image_and_text_page(pdf,os.path.join(output_dir, "distances_batches.png"),title="Distances Batch Plot",
            description="Boxplot comparing Euclidean distances within and between plates.",figsize=(10,10))

        # Page 11: PCA by Factor Plot
        add_image_and_text_page(pdf,os.path.join(output_dir, "pcas_by_factor.png"),title="PCA plots",
            description="PCA colored by hierarchical clustering and shaped by different factors.",figsize=(10,10))

        # Page 12: PCA Plots (Non-imputed vs. Imputed)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16,16))
        axs[0,0].imshow(plt.imread(os.path.join(output_dir, "pca_plot.png")))
        axs[0,0].axis('off')
        axs[0,0].set_title("PCA Plot - 100% Filtering", fontsize=10)
        axs[0,1].imshow(plt.imread(os.path.join(output_dir_imputed, "pca_plot.png")))
        axs[0,1].axis('off')
        axs[0,1].set_title("PCA Plot - Imputed", fontsize=10)
        axs[1,0].imshow(plt.imread(os.path.join(output_dir, "pca_euclidean_distance.png")))
        axs[1,0].axis('off')
        axs[1,0].set_title("100% Filtering", fontsize=10)
        axs[1,1].imshow(plt.imread(os.path.join(output_dir_imputed, "pca_euclidean_distance.png")))
        axs[1,1].axis('off')
        axs[1,1].set_title("Imputed", fontsize=10)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 13: PCA Mahalanobis Distance
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        axs[0].imshow(plt.imread(os.path.join(output_dir, "pca_M_distance.png")))
        axs[0].axis('off')
        axs[0].set_title("100% Filtering", fontsize=10)
        axs[1].imshow(plt.imread(os.path.join(output_dir_imputed, "pca_M_distance.png")))
        axs[1].axis('off')
        axs[1].set_title("Imputed", fontsize=10)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 14: Protein Density and Mean Protein Density (Vertical)
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8.27,11.69))
        ax1.imshow(plt.imread(os.path.join(output_dir, "protein_density_samples.png")))
        ax1.axis('off')
        ax1.set_title("Protein Density per Sample", fontsize=12)
        ax2.imshow(plt.imread(os.path.join(output_dir, "mean_protein_density.png")))
        ax2.axis('off')
        ax2.set_title("Mean Protein Density", fontsize=12)
        pdf.savefig(fig, dpi=300)
        plt.close()

        # Page 15: Mean Protein Histogram
        add_image_and_text_page( pdf, os.path.join(output_dir, "mean_protein_histogram.png"),title="Protein Intensity Distribution b",
            description="Mean protein intensity distribution of proteins, with and without imputation",figsize=(10,8))

        # Page 16: Albumin Scatterplot
        add_image_and_text_page(pdf,os.path.join(output_dir, "albumin_scatterplot.png"),title="Albumin Scatterplot",
            description="Aalbumin intensities across samples with outlier samples annotated.",figsize=(10,8))

        # Page 17: Intra-individual CV Plots (Combined for all conditions)
        add_image_and_text_page(pdf,os.path.join(output_dir, "intra_all.png"),title="Intra-individual CV ",
            description="Intra-individual CV across all individuals.",figsize=(10,8))
        add_image_and_text_page( pdf,os.path.join(output_dir, "intra_by_condition.png"),title="Intra-individual CV by Condition",
            description="Intra-individual by biological condition.",figsize=(10,12))

        # Page 18: Inter-individual CV Plots (Combined for all conditions)
        add_image_and_text_page(pdf,os.path.join(output_dir, "inter_all.png"),title="Inter-individual CV ",
            description="Inter-individual CV across all individuals.",figsize=(10,8))
        add_image_and_text_page( pdf,os.path.join(output_dir, "inter_by_condition.png"),title="Inter-individual CV by Condition",
            description="Inter-individual CVs by biological condition.",figsize=(10,12))
        
        if create_contamination:
            # Page 19: Contamination Panels (Vertical)
            fig, axs = plt.subplots(nrows=2, figsize=(20,16))
            axs[0].imshow(plt.imread(os.path.join(output_dir, "Blood_Contamination_Panel.png")))
            axs[0].axis('off')
            axs[0].set_title("Contamination Panel", fontsize=10)
            axs[1].imshow(plt.imread(os.path.join(output_dir, "Blood_Contamination_Panel_outliers.png")))
            axs[1].axis('off')
            axs[1].set_title("Contamination Panel (Outliers)", fontsize=10)
            pdf.savefig(fig, dpi=300)
            plt.close()
            
        # Page 20: Protein Rank-Abundance Plot
        fig = plt.figure(figsize=(11.69,8.27))
        plt.imshow(plt.imread(rank_plot_path))
        plt.axis('off')
        plt.title("Protein Rank-Abundance Plot", fontsize=14)
        pdf.savefig(fig, dpi=300)
        plt.close()
        
        # Page 21: Outlier Samples Summary Table
        fig = plt.figure(figsize=(12, len(all_outlier_samples)*0.3 + 2))
        plt.axis('off')
        table = plt.table(cellText=table_df.values,colLabels=table_df.columns,rowLabels=table_df.index,cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1,1.5)
        plt.title("Outlier Samples Summary\nOutliers were identified using a 99% confidence interval", fontsize=14)
        pdf.savefig(fig, dpi=300)
        plt.close()
