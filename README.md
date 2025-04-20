# Proteomics Quantification and Quality Control Pipeline

This **Snakemake** pipeline allows proteomics researchers to **quantify large-cohort DIA mass spectrometry data** and **automatically generate a quality control (QC) report in PDF format**. It integrates preprocessing, filtering, normalization, PCA, outlier detection, and contamination analysis into a reproducible and scalable workflow.

---

## Installation

Clone the repository with submodules:

```bash
git clone --recursive git@github.com:......git
```

After cloning, configure your pipeline by editing the configuration file.

---

## Configuration

Edit the config file (YAML format) with the appropriate paths and options:

```yaml
# Path to the root directory of the pipeline
project_root: '.../pipeline/'

# Path to the metadata CSV file 
metadata: ".../pipeline/data_input/metadata_file.csv"

# Path to the FASTA file containing the reference protein sequence database (used for spectral library generation)
fasta_file: '.../pipeline/data_input/fasta_file'

# Filtering settings for Quality Control
# Options:
#"all" (all samples),
#"group" (at least one group)
filtering_option: all
# Specify group for filtering when using filtering_option: group
group: None

#Filtering percentage - default 20%
filtering_percentage: 0.20

# Metadata column for grouping in missing value analysis
missing_values_grouping: 'condition'

# Metadata columns used for PCA visualization
pca_factors: ["plate", "condition", "time"]

# Enable or disable the contamination panel analysis (True or False)
contamination_panel: True

# Path to the contamination panel Excel file
contamination_panel_file: "/.../pipeline/data_input/contamination_panel.xlsx"

#Indicate the samples to exclude, in the format of a list ["sxxx", "sxxx"]
exclude_samples: []

# Batch correction settings
batch_correction: False
# Metadata column indicating the batch factor
batch_effect_column: "plate"
```


---

## Usage

To run the full workflow and generate a QC report, execute:

```bash
run slurm.sh all_workflow
```

This will perform quantification, generate QC plots, and output a full PDF report.

---

### Post-Quality Control

After analysing the report, you may want to:

#### Apply batch correction

```bash
run slurm.sh batch_correction
```

#### Remove identified outlier samples

```bash
run slurm.sh remove_samples
```

---

## Output

The pipeline generates:
- **Quality Control visualizations in png** 
- A **comprehensive PDF report** 

---


