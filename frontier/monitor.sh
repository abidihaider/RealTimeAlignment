#!/bin/bash

num_epochs=200
job_id=$1

JOB_DIR="job_roots/job_root_$job_id"
JOB_NAME="rtal_hp-job_root_$job_id"
ACCOUNT_USER=$(whoami)

while true; do
    # Check for running/pending SLURM jobs with the name
    JOB_FOUND=$(squeue -u "$ACCOUNT_USER" -o "%.100j" -n "$JOB_NAME" | grep "$JOB_NAME")

    if [ -z "$JOB_FOUND" ]; then
        # No job found; check file numbers
        MIN=999999  # Start with a large number
        for folder in "$JOB_DIR"/mlp*; do
            fname="${folder}/checkpoints/last_saved_epoch"
            if [[ -f "$fname" ]]; then
                VAL=$(<"$fname")
                if [[ "$VAL" =~ ^[0-9]+$ ]]; then
                    (( VAL < MIN )) && MIN=$VAL
                fi
            fi
        done

        if (( MIN < num_epochs )); then
            echo "Submitting $JOB_NAME since min epoch ($MIN) < $num_epochs"
            (
                cd "$JOB_DIR" || exit
                command="sbatch submit.sh"
                # echo "$command"
                ${command}
            )
        else
            echo "Job $JOB_NAME is done!"
        fi
    else
        echo "Job '$JOB_NAME' is already running or queued."
    fi

    sleep 60
done
