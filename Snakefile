configfile: "config/config.yml"

include: "rules/A_diann.smk"
include: "rules/B_QC.smk"
include: "rules/C_diann_remove.smk"
include: "rules/D_spectronaut.smk"

directory_root = config["project_root"]

        
