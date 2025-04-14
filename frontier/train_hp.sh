#SBATCH --job-name=hyperparam_tuning_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --array=0-24
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

# Load modules or conda environments as needed
module load cuda/11.7  # or your appropriate module
# source activate myenv

CONFIG_DIR=configs
TOTAL_CONFIGS=200
CONFIGS_PER_NODE=8

START_INDEX=$((SLURM_ARRAY_TASK_ID * CONFIGS_PER_NODE))
END_INDEX=$((START_INDEX + CONFIGS_PER_NODE - 1))

echo "Running on node $SLURM_NODEID (array task $SLURM_ARRAY_TASK_ID) for configs $START_INDEX to $END_INDEX"

for i in $(seq 0 $((CONFIGS_PER_NODE - 1))); do
    CONFIG_INDEX=$((START_INDEX + i))
    CONFIG_FILE=$(printf "%s/config_%03d.yaml" "$CONFIG_DIR" "$CONFIG_INDEX")
    GPU_ID=$i

    if [ -f "$CONFIG_FILE" ]; then
        echo "Launching config $CONFIG_FILE on GPU $GPU_ID"
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py --config "$CONFIG_FILE" &
    else
        echo "Config file $CONFIG_FILE not found, skipping."
    fi
done

wait

