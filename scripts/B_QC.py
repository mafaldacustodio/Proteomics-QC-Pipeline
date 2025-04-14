################################ IMPORTS #########################
import alphastats, os, sys, argparse, io, contextlib
import pandas as pd
import snakemake 
sys.path.append("/projects/cbmr_fpm_soup-AUDIT/data/pipeline/scripts")
from RobustPCA import RobustPCA
from module import *
import copy 
################################# DIRECTORIES ##########################################
# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument("--matrix", required=True)
parser.add_argument("--metadata", required=True)
parser.add_argument("--contamination", required=True)
parser.add_argument("--outputMatrix", required=True)
parser.add_argument("--outputReport", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--filtering_option", required=True)
parser.add_argument("--group", required=True)
parser.add_argument("--filtering_percentage", type=float, required=True)
parser.add_argument("--pca_factors", required=True)
parser.add_argument("--missing_values_group", required=True)
parser.add_argument("--contamination_panel_flag", required=True)
parser.add_argument("--exclude_samples", required=True)
parser.add_argument("--batch_correction", required=True)
parser.add_argument("--batch_correction_column", required=True)
args = parser.parse_args()

# Input
pg_matrix = args.matrix
metadata_path = args.metadata
contamination_panel = args.contamination
create_contamination_panel = args.contamination_panel_flag
filtering_option = args.filtering_option
filtering_percentage = args.filtering_percentage
factors = args.pca_factors.split()
grp = args.group
missing_values_group = args.missing_values_group
batch_correction = args.batch_correction
batch_correction_column = args.batch_correction_column

# Output
pg_matrix_new  = args.outputMatrix
report = args.outputReport
output_dir = args.output

# Extract workflow and cohort from paths
workflow_cohort = os.path.basename(pg_matrix).replace(".pg_matrix.tsv", "")
workflow = workflow_cohort.split("_")[0]  # Extract workflow
cohort = workflow_cohort.split("_")[1] if "_" in workflow_cohort else None

# Create directories if they don't exist
output_dir_imputed = f"{output_dir}/data_imputed"
os.makedirs(output_dir_imputed, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

######################################## CORRECT COLUMNS NAME (from pg matrix) ###########################

rename_mzML_columns(pg_matrix, pg_matrix_new)

################################################ READ METADATA ####################################################
# Filter metadata for respective workflow/cohort only
metadata = pd.read_csv(metadata_path, delimiter=";")
metadata.columns = metadata.columns.str.strip()
if cohort is None:
    metadata = metadata[(metadata['workflow'] == workflow)]
else:
    metadata = metadata[(metadata['workflow'] == workflow) & (metadata['cohort'] == cohort)]

########################################## ALPHAPEPSTATS LOAD DATA #############################################

diann_data = alphastats.DIANNLoader(file=pg_matrix_new)

ds = alphastats.DataSet(loader=diann_data, metadata_path=metadata, sample_column="analytical_id")
ds_imputed = alphastats.DataSet(loader=diann_data, metadata_path=metadata, sample_column="analytical_id")
ds_rawIntensity = alphastats.DataSet(loader=diann_data, metadata_path=metadata, sample_column="analytical_id")

################################################ PREPROCESS ##############################################
# Remove contaminations and perform log2 transformation 
ds.preprocess(remove_contaminations=True)
# Remove contaminations and perform log2 transformation and imputation
ds_imputed.preprocess(remove_contaminations=True, imputation="knn")
# Remove contaminations and no log2 transformation
ds_rawIntensity.preprocess(remove_contaminations=True, log2_transform =False)

ds_orig = copy.deepcopy(ds)

# Capture pre–process info text 
with io.StringIO() as f, contextlib.redirect_stdout(f):
    ds.preprocess_print_info()
    preprocess_info_text = f.getvalue()

num_proteins_before = ds.mat.shape[1]

generate_barplot_thresholds(ds, output_dir)

# Filtering for 20%
if filtering_option == 'all':
    ds = filter_proteins_by_threshold(ds, filtering_percentage)
    ds_rawIntensity = filter_proteins_by_threshold(ds_rawIntensity, filtering_percentage)
    # Filtering for 100% (for ds_without_NA)
    ds_without_NA = filter_proteins_by_threshold(ds_orig, 1.0)
else:
    ds = select_grps(ds, grp, filtering_percentage)
    ds_rawIntensity = select_grps(ds_rawIntensity, grp, filtering_percentage)
    # Filtering for 100% (for ds_without_NA)
    ds_without_NA = select_grps(ds_orig, grp, 1.0)

num_proteins_after = ds.mat.shape[1]

# Normalization (median scaling)
ds.mat = median_scaling(ds.mat.T, scale=True)
ds_imputed.mat = median_scaling(ds_imputed.mat.T, scale=True)
ds_without_NA.mat = median_scaling(ds_without_NA.mat.T, scale=True)
ds_rawIntensity.mat = median_scaling(ds_rawIntensity.mat.T, scale=True)

if batch_correction:
    ds_imputed.batch_correction(batch_correction_column)
    ds_without_NA.batch_correction(batch_correction_column)

######################################## QC PLOTS ######################################

#Sample distribution after preprocess with outliers highlighted
outliers_distribution = generate_sample_distribution_plot(ds, output_dir)

#Missing values Heatmap 
plot_missing_values_heatmap(ds.mat, output_dir)

#Missing values per group
plot_missing_values_per_group(ds, missing_values_group, output_dir)

#missing proteins by group
plot_missing_proteins_distribution_by_group(ds, missing_values_group, output_dir)

#Sample Correlation Heatmap
plot_sample_correlation_heatmap(ds.mat.T, output_dir)

#dendogram
plot_hierarchical_clustering(ds_imputed.mat, output_dir, ds_imputed.metadata, factors)

#PCA colored by cluster and shaped by factor
n_clusters = ds_imputed.metadata["plate"].nunique()
plot_all_pcas_by_factor(ds_imputed.mat,ds_imputed.metadata, factors, n_clusters,output_dir)

#boxplot of distances within and between plates
out_path = plot_plate_distances(ds_imputed.mat, metadata, output_dir)

#PCA Plots
# Non impuited PCA
pca_non_imputed = RobustPCA(dataset=ds_without_NA, group='plate', circle=True, output_dir=output_dir)
# Imputed PCA
pca_imputed = RobustPCA(dataset=ds_imputed, group='plate', circle=True, output_dir=output_dir_imputed)

pca_euclidean_samples = f"{pca_non_imputed.top5_euclidean}"
pca_mahanobis_samples = f"{pca_non_imputed.top5_mahalanobis}"
pca_imputed_euclidean_samples = f"{pca_imputed.top5_euclidean}"
pca_imputed_mahalanobis_samples = f"{pca_imputed.top5_mahalanobis}"

protein_map = dict(zip(ds_imputed.rawinput["Protein.Group"], ds_imputed.rawinput["Protein.Names"]))
plot_enrichment_pcs(ds_imputed.mat, output_dir,20, protein_map)


#Density plot of samples' intensities
density_outliers = plot_density_per_sample(ds, output_dir)

#Mean intensity density plot, with outliers samples 
plot_mean_intensity_density(ds, output_dir, density_outliers)

#Histogram of mean intensity of non-imputed vs imputed data
plot_intensity_histogram_with_imputed(ds, ds_imputed, output_dir)

#ALBUMIN CONCENTRATION
out_albumin, albumin_outliers_str = plot_albumin_concentration(ds, output_dir)

#Protein Rank Plot
rank_path = plot_protein_rank_abundance(ds, output_dir)

### CV 
#Get protein order from rank-abundance analysis.
protein_order = compute_protein_rank_order(ds_rawIntensity)
#plot Intra-individual CVs:
plot_cv_scatter(compute_intra_cv(ds_rawIntensity), protein_order, "Intra-individual CV (All)", os.path.join(output_dir, "intra_all.png"))
plot_cv_scatter_by_conditions(ds_rawIntensity, "condition", protein_order, compute_intra_cv, os.path.join(output_dir, "intra_by_condition.png"))
#plot Inter-individual CVs:
plot_cv_scatter(compute_inter_cv(ds_rawIntensity), protein_order, "Inter-individual CV (All)", os.path.join(output_dir, "inter_all.png"))
plot_cv_scatter_by_conditions(ds_rawIntensity, "condition", protein_order, compute_inter_cv,os.path.join(output_dir, "inter_by_condition.png"))

## Contamination Panel ###
if create_contamination_panel:
    contamination_data = prepare_data_contamination(pg_matrix_new, contamination_panel)
    plot_contamination_panel(contamination_data, output_dir, highlight_outliers=False)
    plot_contamination_panel(contamination_data, output_dir, highlight_outliers=True)
    outlier_info = plot_contamination_panel(contamination_data, output_dir, highlight_outliers=True)
else:
    outlier_info = ""
########################################## CREATE CSV FILES #########################################

summary_data_dir = os.path.join(output_dir, "summary_data")
os.makedirs(summary_data_dir, exist_ok=True)

# 1. Save Number of Proteins Data
num_proteins_df = pd.DataFrame({
    "num_proteins_before": [num_proteins_before],
    "num_proteins_after": [num_proteins_after]})
num_proteins_df.to_csv(os.path.join(summary_data_dir, "num_proteins.csv"), index=False)

# 2. Save CV Values
stats = ds.mat.describe()
cv_values = np.sqrt(np.exp(stats.loc["std"]**2) - 1) * 100 
cv_df = cv_values.reset_index()
cv_df.columns = ["protein", "cv_values"]
cv_df.to_csv(os.path.join(summary_data_dir, "cv_values.csv"), index=False)

# 3. Save Albumin Concentration Data
albumin_id="P02768"
albumin_values = ds_rawIntensity.mat[albumin_id]
sample_labels = ds_rawIntensity.mat.index
total_intensity = ds_rawIntensity.mat.sum(axis=1) 
albumin_df = pd.DataFrame({"Sample": sample_labels,"Albumin_Concentration":albumin_values, "Total_Intensity": total_intensity})
albumin_df.to_csv(os.path.join(summary_data_dir, "albumin_concentration.csv"), index=False)

# 4. Save normalized protein matrix 
ds.mat.to_csv(os.path.join(summary_data_dir, "normalized_protein_matrix.csv"))


######################################### GENERATE PDF REPORT #########################################

generate_pdf_report(report, output_dir, output_dir_imputed, workflow_cohort, num_proteins_before, num_proteins_after, 
                        outliers_distribution, pca_euclidean_samples, pca_mahanobis_samples,pca_imputed_euclidean_samples, pca_imputed_mahalanobis_samples,
                        albumin_outliers_str,create_contamination_panel, outlier_info, density_outliers, rank_path)
