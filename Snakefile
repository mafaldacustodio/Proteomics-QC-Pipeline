configfile: "config/config.yml"

include: "rules/A_diann.smk"
include: "rules/B_QC.smk"

directory_root = config["project_root"]

rule all_workflow:
    input:
        f"{directory_root}data_output/library.predicted.speclib",
        f"{directory_root}data_output/A_all_converted.marker",
        f"{directory_root}data_output/A_all_complete.marker",
        directory(f"{directory_root}data_output/QC/"),
        f"{directory_root}data_output/B_QC_individual_complete.marker",
        f"{directory_root}data_output/B_QC_summary_complete.marker"

rule all_remove_samples:
    input:
        f"{directory_root}data_output/A_all_filtered_complete.marker",
        directory(f"{directory_root}data_output/QC_remove_samples/"),
        f"{directory_root}data_output/B_QC_individual_complete.marker",
        f"{directory_root}data_output/B_QC_summary_complete.marker"

rule all_batch_correction:
    input:
        directory(f"{directory_root}data_output/QC_batch_correction/")

        