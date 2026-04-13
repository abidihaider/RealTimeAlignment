#!/bin/bash
#
# Set conda + project environment. From a login shell:
#   source /gpfs/mnt/gpfs01/usfcc/pusharma/Haider_LDRD/RealTimeAlignment/setEnv.sh
#

export REALTIME_ALIGNMENT_ROOT=/gpfs/mnt/gpfs01/usfcc/pusharma/Haider_LDRD/RealTimeAlignment

# source /cvmfs/sft.cern.ch/lcg/views/dev4cuda/latest/x86_64-centos7-gcc11-opt/setup.sh 
# source /cvmfs/sft.cern.ch/lcg/views/LCG_105a_cuda/x86_64-centos7-gcc11-opt/setup.sh


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
hostname=$(hostname)
echo "Hostname: $hostname"

# Check if the hostname contains "bnl"
if [[ $hostname == *"bnl"* ]]; then
 __conda_setup="$('/usatlas/u/pusharma/atlasdisk/miniconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/usatlas/u/pusharma/atlasdisk/miniconda/etc/profile.d/conda.sh" ]; then
            . "/usatlas/u/pusharma/atlasdisk/miniconda/etc/profile.d/conda.sh"
        else
            export PATH="/usatlas/u/pusharma/atlasdisk/miniconda/bin:$PATH"
        fi
    fi
else
    if [[ $hostname == *"zephyr"* ]]; then
        __conda_setup="$('/home/zephyr/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
        if [ $? -eq 0 ]; then
            eval "$__conda_setup"
        else
            if [ -f "/home/zephyr/miniconda3/etc/profile.d/conda.sh" ]; then
                . "/home/zephyr/miniconda3/etc/profile.d/conda.sh"
            else
                export PATH="/home/zephyr/miniconda3/bin:$PATH"
            fi
        fi
    else
        __conda_setup="$('/data/pusharma/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
        if [ $? -eq 0 ]; then
            eval "$__conda_setup"
        else
            if [ -f "/data/pusharma/miniconda3/etc/profile.d/conda.sh" ]; then
                . "/data/pusharma/miniconda3/etc/profile.d/conda.sh"
            else
                export PATH="/data/pusharma/miniconda3/bin:$PATH"
            fi
        fi
    fi    
fi

unset __conda_setup
# <<< conda initialize <<<
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH

# conda activate mltrack_tf2p18
conda activate qat
conda info --envs

# nvidia-smi


export PYTHONNOUSERSITE=1
export DATAROOT=/gpfs/mnt/gpfs01/usfcc/pusharma/Haider_LDRD/rom_det-3_part-200_cont-and-rounded
echo "REALTIME_ALIGNMENT_ROOT=$REALTIME_ALIGNMENT_ROOT"
echo "DATAROOT is set to: $DATAROOT"
echo "Environment set!"

# python train.py --config config.yaml --gpu-id 0
