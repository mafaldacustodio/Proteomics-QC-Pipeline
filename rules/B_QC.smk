import pandas as pd

directory_root = config["project_root"]

# Define workflows and cohorts before using them
metadata = pd.read_csv(f"{directory_root}data_input/ms_raw_files.csv", delimiter=";")
metadata.columns = metadata.columns.str.strip()

# List of unique workflows and cohorts (if needed)
workflows = metadata["workflow"].unique()
cohorts = sorted(metadata["cohort"].unique())


if config["batch_correction"] == False and config["exclude_samples"] == []:
    qc_version = "QC"
elif config["batch_correction"] == True:
    qc_version = "QC_batch_correction"
else:
    qc_version = "QC_remove_samples"

ruleorder: run_alphastats_analysis_cohort > run_alphastats_analysis_workflow

rule setup_env:
    output:
        env_flag=f"{directory_root} .env_setup"
    shell:
        """
        module purge
        uv venv --python 3.9
        uv pip install venn
        uv pip install alphastats
        uv pip install adjustText
        uv pip install snakemake
        touch {output.env_flag}
        """

rule run_alphastats_analysis_workflow:
    input:
        matrix= f"{directory_root}data_output/{{workflow}}/{{workflow}}.pg_matrix.tsv"
    output:
        new_matrix=f"{directory_root}data_output/{{workflow}}/{{workflow}}_corrected.pg_matrix.tsv",
        output_dir = directory(f"{directory_root}data_output/{qc_version}/{{workflow}}"),
        report = f"{directory_root}data_output/{qc_version}/{{workflow}}/{{workflow}}_report.pdf"
    threads: 16
    resources:
        mem_mb = 262144,
        slurm_partition = "standardqueue"
    shell:
        """
        uv run {directory_root}scripts/B_QC.py --matrix {input.matrix} \
      --metadata  {config[metadata]} \
      --contamination {config[contamination_panel_file]} \
      --outputMatrix {output.new_matrix} \
      --outputReport {output.report} \
      --output {output.output_dir} \
      --filtering_option {config[filtering_option]} \
      --group {config[group]} \
      --filtering_percentage {config[filtering_percentage]} \
      --pca_factors '{config[pca_factors]}' \
      --missing_values_group '{config[missing_values_grouping]}' \
      --contamination_panel_flag {config[contamination_panel]} \
      --exclude_samples '{config[exclude_samples]}' \
      --batch_column {config[batch_column]}  \
      --batch_correction {config[batch_correction]}  \
      --batch_correction_column {config[batch_effect_column]}
        """


rule run_alphastats_analysis_cohort:
    input:
        matrix=f"{directory_root}data_output/{{workflow}}_{{cohort}}/{{workflow}}_{{cohort}}.pg_matrix.tsv"
    output:
        new_matrix=f"{directory_root}data_output/{{workflow}}_{{cohort}}/{{workflow}}_{{cohort}}_corrected.pg_matrix.tsv",
        output_dir = directory(f"{directory_root}data_output/{qc_version}/{{workflow}}_{{cohort}}"),
        report = f"{directory_root}data_output/{qc_version}/{{workflow}}_{{cohort}}/{{workflow}}_{{cohort}}_report.pdf"
    threads: 16
    resources:
        mem_mb=262144,
        slurm_partition="standardqueue"
    shell:
        """
        uv run {directory_root}scripts/B_QC.py --matrix {input.matrix} \
              --metadata  {config[metadata]} \
      --contamination {config[contamination_panel_file]} \
      --outputMatrix {output.new_matrix} \
      --outputReport {output.report} \
      --output {output.output_dir} \
      --filtering_option {config[filtering_option]} \
      --group {config[group]} \
      --filtering_percentage {config[filtering_percentage]} \
      --pca_factors '{config[pca_factors]}' \
      --missing_values_group '{config[missing_values_grouping]}' \
      --contamination_panel_flag {config[contamination_panel]} \
      --exclude_samples '{config[exclude_samples]}' \
      --batch_column {config[batch_column]}  \
      --batch_correction {config[batch_correction]} \
      --batch_correction_column {config[batch_effect_column]}
        """ 

rule run_all_QC:
    input:
        expand(f"{directory_root}data_output/{{workflow}}/{{workflow}}_corrected.pg_matrix.tsv",
               workflow=workflows),
        expand(f"{directory_root}data_output/{{workflow}}_{{cohort}}/{{workflow}}_{{cohort}}_corrected.pg_matrix.tsv",
               workflow=workflows, cohort=cohorts)
    output:
        marker = f"{directory_root}data_output/B_QC_individual_complete.marker"
    shell:
        "touch {output.marker}"


rule summarize_qc:
    input:
        qc_root = f"{directory_root}data_output/QC/"
    output:
        summary_plots = directory(f"{directory_root}data_output/QC/summary_plots"),
        marker = f"{directory_root}data_output/B_QC_summary_complete.marker"
    resources:
        mem_mb=262144,
        slurm_partition="standardqueue"
    shell:
        """
        uv run {directory_root}scripts/summary_report.py 
        touch {output.marker}
        """


