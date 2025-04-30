#!/bin/bash

#SBATCH -A LRN057
#SBATCH -p batch
#SBATCH --job-name=rtal_hp-_SUBDIR_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm-%A_%a.o
#SBATCH --error=logs/slurm-%A_%a.e

# Load modules
module load PrgEnv-gnu
module load emacs
module load rocm/6.2.4
module load gcc/12.2.0

# Activate environment
source /lustre/orion/lrn057/world-shared/miniconda3-frontier/bin/activate ""
conda activate rtdc

# project folder
PROJ_ROOT=/lustre/orion/lrn057/scratch/yhuang2/rtal

TRAIN_SCRIPT="${PROJ_ROOT}/RealTimeAlignment/train/mlp/train.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo -e "train script $TRAIN_SCRIPT does not exist!"
    exit 1
fi

JOB_ROOT="${PROJ_ROOT}/RealTimeAlignment/frontier/job_roots/_SUBDIR_"
if [ ! -d "$JOB_ROOT" ]; then
    echo -e "job root $JOB_ROOT does not exist!"
    exit 1
fi

DATA_ROOT="${PROJ_ROOT}/rom_det-3_part-200_cont-and-rounded/"
if [ ! -d "$DATA_ROOT" ]; then
    echo -e "data root $DATA_ROOT does not exist!"
    exit 1
fi

JOBS_PER_NODE=8

# Get a sorted list of all job subfolders (as an array)
mapfile -t JOB_DIRS < <(find "$JOB_ROOT" -maxdepth 1 -type d -name 'mlp|*'| sort)

# note
START_INDEX=$((SLURM_ARRAY_TASK_ID * JOBS_PER_NODE))
END_INDEX=$((START_INDEX + JOBS_PER_NODE - 1))
echo "Task ID $SLURM_ARRAY_TASK_ID running jobs from index $START_INDEX to $END_INDEX"

for i in $(seq 0 $((JOBS_PER_NODE - 1))); do
    JOB_INDEX=$((START_INDEX + i))
    JOB_DIR="${JOB_DIRS[$JOB_INDEX]}"
    GPU_ID=$i

    if [ -d "$JOB_DIR" ]; then
        echo "Launching job in $JOB_DIR on GPU $GPU_ID"
        (
            cd "$JOB_DIR" || exit
            export DATAROOT="$DATA_ROOT"
            cmd="python ${TRAIN_SCRIPT} --config config.yaml --gpu-id $GPU_ID"
            ${cmd}
        ) &
    else
        echo "Job directory $JOB_DIR not found or invalid, skipping."
    fi
done

wait
