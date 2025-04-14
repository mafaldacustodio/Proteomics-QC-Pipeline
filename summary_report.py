import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# Define input directory containing all QC summary files
summary_root = "/projects/cbmr_fpm_soup-AUDIT/data/pipeline/data_output/QC/"
workflow_cohort_dirs = glob.glob(os.path.join(summary_root, "*", "summary_data"))

# Create output directory for summary plots and report
summary_output_dir = os.path.join(summary_root, "summary_plots")
os.makedirs(summary_output_dir, exist_ok=True)


num_proteins_data, cv_data, alb_data, protein_matrices = [], [], [], {}

# Read data from each workflow cohort
for qc_dir in workflow_cohort_dirs:
    workflow_cohort = os.path.basename(os.path.dirname(qc_dir))
    # Read number of proteins data
    num_proteins_df = pd.read_csv(os.path.join(qc_dir, "num_proteins.csv"))
    num_proteins_df["workflow_cohort"] = workflow_cohort
    num_proteins_data.append(num_proteins_df)
    # Read CV values data
    cv_df = pd.read_csv(os.path.join(qc_dir, "cv_values.csv"), index_col=0)
    cv_df["workflow_cohort"] = workflow_cohort
    cv_data.append(cv_df)
    # Read albumin concentration data (if needed)
    alb_df = pd.read_csv(os.path.join(qc_dir, "albumin_concentration.csv"), index_col=0)
    alb_df["workflow_cohort"] = workflow_cohort
    alb_data.append(alb_df)

    mat_path = os.path.join(qc_dir, "normalized_protein_matrix.csv")
    df = pd.read_csv(mat_path, index_col=0)
    protein_matrices[workflow_cohort] = df.T

# Combine data frames for protein count and CV plots
num_proteins_df = pd.concat(num_proteins_data, ignore_index=True)
cv_df = pd.concat(cv_data, ignore_index=True)
alb_df = pd.concat(alb_data, ignore_index=True)


# Plot 1: Protein Count Across Workflow-Cohorts
melted_df = num_proteins_df.melt(id_vars=["workflow_cohort"],
                                 value_vars=["num_proteins_before", "num_proteins_after"],
                                 var_name="filter_status", value_name="protein_count")
workflow_order = sorted(melted_df['workflow_cohort'].unique())
plt.figure(figsize=(12, 6))
sns.barplot(x="workflow_cohort", y="protein_count", hue="filter_status", data=melted_df,order=workflow_order)
plt.xticks(rotation=45)
plt.ylabel("Number of Proteins")
plt.ylabel("Number of Proteins")
plt.title("Protein Count Across Workflow (Before and After Filtering)")
plt.tight_layout()
num_proteins_plot_path = os.path.join(summary_output_dir, "num_proteins_comparison.png")
plt.savefig(num_proteins_plot_path, dpi=300)
num_proteins_fig = plt.gcf()  # save current figure for PDF
plt.close()



# Compute Albumin Percentage
alb_df["albumin_percent"] = (alb_df["Albumin_Concentration"] / alb_df["Total_Intensity"]) * 100

# Plot 3: Albumin % Scatterplots 
workflow_cohort_names = sorted(alb_df['workflow_cohort'].unique())
albumin_pages = []
plots_per_page = 6
ncols, nrows = 3, 2

for i in range(0, len(workflow_cohort_names), plots_per_page):
    sub_cohorts = workflow_cohort_names[i:i + plots_per_page]
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 10))
    axs = axs.flatten()

    for j, cohort in enumerate(sub_cohorts):
        ax = axs[j]
        sub_df = alb_df[alb_df["workflow_cohort"] == cohort]
        alb_percent = sub_df["albumin_percent"].values
        sample_labels = sub_df.index.astype(str)

        sns.scatterplot(x=sample_labels, y=alb_percent, color="red", s=40, alpha=0.7, ax=ax)
        ax.set_title(f"Albumin % - {cohort}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Albumin / Total (%)")
        ax.set_xticks([])
        ax.set_ylim(0, 100)

    for k in range(j + 1, plots_per_page):
        axs[k].axis("off")

    fig.tight_layout()
    albumin_pages.append(fig)

###################################### correlation matrix  ####################### 
# Keep only workflow names without underscores (i.e., "singular" names)
workflow_names = [wf for wf in protein_matrices.keys() if "_" not in wf]
print(workflow_names)
for i in range(len(workflow_names)):
    for j in range(i + 1, len(workflow_names)):
        wf1, wf2 = workflow_names[i], workflow_names[j]
        df1, df2 = protein_matrices[wf1], protein_matrices[wf2]

        # Find shared proteins
        shared_proteins = df1.index.intersection(df2.index)
        if len(shared_proteins) == 0:
            print(f"No shared proteins between {wf1} and {wf2}")
            continue

        # Find shared proteins
        shared_proteins = df1.index.intersection(df2.index)
        if len(shared_proteins) == 0:
            print(f"No shared proteins between {wf1} and {wf2}")
            continue

        # Extract shared proteins (already log2-transformed)
        df1_shared = df1.loc[shared_proteins]
        df2_shared = df2.loc[shared_proteins]

        # Ensure proteins are aligned and samples are columns
        df1_shared = df1_shared
        # Find shared proteins
        shared_proteins = df1.index.intersection(df2.index)
        if len(shared_proteins) == 0:
            print(f"No shared proteins between {wf1} and {wf2}")
            continue

        df1_shared = df1.loc[shared_proteins]
        df2_shared = df2.loc[shared_proteins]

        # Combine into one matrix: columns = samples from both workflows
        combined = pd.concat([df1_shared, df2_shared], axis=1)

        # Compute correlation between all samples across shared proteins
        corr_matrix = combined.corr(method="spearman")

        # Plot and save the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="vlag")
        plt.title(f"Sample Correlation: {wf1} vs {wf2}\n(Shared proteins: {len(shared_proteins)})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_path = os.path.join(summary_output_dir, f"workflow_sample_correlation_{wf1}_vs_{wf2}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()





############################ protein abundance rank ############################
plt.figure(figsize=(10, 6))

workflow_names = [wf for wf in protein_matrices.keys() if "_" not in wf]

for wf in workflow_names:
    df = protein_matrices[wf]
    # Compute mean intensity per protein across all samples in the workflow
    mean_protein_abundance = df.T.mean(axis=0).sort_values(ascending=False)
    # Rank proteins and plot
    sns.scatterplot(x=range(1, len(mean_protein_abundance) + 1),y=mean_protein_abundance.values,s=40,label=wf, edgecolor='grey',linewidth=0.3)

plt.xlabel("Protein Abundance Rank")
plt.ylabel("Log2 Intensity")
plt.title("Protein Abundance Rank Plot")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
protein_rank= os.path.join(summary_output_dir, "protein_rank_abundance_by_workflow.png")
plt.savefig(protein_rank, dpi=300)
protein_rank_plot = plt.gcf() 
plt.close()

####################################################################################
from venn import venn

# Prepare sets
workflow_names = [wf for wf in protein_matrices if "_" not in wf]
protein_sets = {wf: set(protein_matrices[wf].index) for wf in workflow_names}
print(protein_sets)
# Optional custom colors (or use a colormap name like 'Set1' or 'tab10')
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # blue, orange, green, red

# Plot with customization
plt.figure(figsize=(12, 9))
venn(
    protein_sets,
    fmt="{size}",           # just show counts
    cmap=colors[:len(protein_sets)],  # match number of workflows
    alpha=0.6,
    fontsize=10,
    legend_loc="upper right"
)
plt.title("Protein Group Overlap Between Workflows", fontsize=14)
plt.tight_layout()

# Save
venn_path = os.path.join(summary_output_dir, f"protein_venn_{len(protein_sets)}sets.png")
plt.savefig(venn_path, dpi=300)
venn_fig = plt.gcf()
plt.close()


########################
# Write All Plots to PDF
report_pdf = os.path.join(summary_output_dir, "QC_report.pdf")
with PdfPages(report_pdf) as pdf:
    # Title Page
    fig_title = plt.figure(figsize=(8.27, 11.69))
    plt.axis('off')
    plt.text(0.5, 0.5, "QC Report for Proteomics Data Analysis",
             ha='center', va='center', fontsize=24)
    pdf.savefig(fig_title)
    plt.close(fig_title)
    
    # Page 3: Protein Count Bar Plot
    pdf.savefig(num_proteins_fig)

    pdf.savefig(protein_rank_plot)

    pdf.savefig(venn_fig)

    for fig in albumin_pages:
        pdf.savefig(fig)
        plt.close(fig)

#         # Add Workflow-vs-Workflow Correlation Heatmaps
    correlation_pngs = sorted([
        f for f in os.listdir(summary_output_dir)
        if f.startswith("workflow_sample_correlation_") and f.endswith(".png")
    ])
 
    for png_file in correlation_pngs:
        fig = plt.figure(figsize=(11.69, 8.27))  # Landscape A4
        img = plt.imread(os.path.join(summary_output_dir, png_file))
        plt.imshow(img)
        plt.axis('off')
        title_str = png_file.replace("workflow_sample_correlation_", "").replace(".png", "").replace("_", " vs ")
        plt.title(f"Workflow Correlation: {title_str}", fontsize=14)
        pdf.savefig(fig)
        plt.close()

    



