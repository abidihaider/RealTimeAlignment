#!/bin/bash

#SBATCH -A LRN057
#SBATCH -p batch
#SBATCH --job-name=test_signal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=00:01:00
#SBATCH --output=logs/test_signal-%A.o
#SBATCH --error=logs/test_signal-%A.e
#SBATCH --signal=TERM@30

# Load modules
module load PrgEnv-gnu
module load emacs
module load rocm/6.2.4
module load gcc/12.2.0

# Activate environment
source /lustre/orion/lrn057/world-shared/miniconda3-frontier/bin/activate ""
conda activate rtdc

# Function to forward SIGTERM to Python
cleanup() {
    echo "Bash trap caught SIGTERM — forwarding to child"
    kill -TERM "$child" 2>/dev/null
}

# Set up trap for SIGTERM
trap cleanup SIGTERM

# Run your Python script in background so we can track its PID
python test.py &
child=$!

# Wait for Python process to finish
wait "$child"

echo "Script finished."
