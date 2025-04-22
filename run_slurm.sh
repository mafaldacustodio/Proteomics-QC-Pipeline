#!/ibin/bash
module load miniconda/24.5.0
module load snakemake/7.30.1
snakemake                                                            \
    --slurm                                                          \
    --default-resources slurm_partition=standardqueue                \
    --rerun-triggers mtime                                           \
    --jobs 32                                                        \
    --cluster-config "config/slurm.yml"                              \
    --configfile config/config.yml                                   \
    --keep-going `# Go on with independent jobs if a job fails.`     \
    --latency-wait 60                                                \
    $target
