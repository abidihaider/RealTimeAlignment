#!/bin/bash

TEMPLATE_FILE="train_hp.sh"
OUTPUT_FILE="replaced.sh"

# Copy the template to output file to start replacements
cp "$TEMPLATE_FILE" "$OUTPUT_FILE"

subdir=$1
# Loop through the arguments
sed -i "s|_SUBDIR_|${subdir}|g" "$OUTPUT_FILE"

PROJ_ROOT=/lustre/orion/lrn057/scratch/yhuang2/rtal
JOB_ROOT="${PROJ_ROOT}/RealTimeAlignment/frontier/job_roots/${subdir}"
if [ ! -d "$JOB_ROOT" ]; then
    echo -e "job root $JOB_ROOT does not exist!"
    exit 1
fi

mv "$OUTPUT_FILE"  "${JOB_ROOT}/submit.sh"

# Submit with sbatch
(
    cd "$JOB_ROOT" || exit
    command="sbatch submit.sh"
    ${command}
)
