#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos kostas-high
#SBATCH --partition kostas-compute
# SBATCH --signal=B:USR1@120

#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=8
# SBATCH --nodes 1
# SBATCH --exclude node-v100-0,node-1080ti-0,node-1080ti-1,node-1080ti-2
# SBATCH --exclude node-v100-0,node-2080ti-4
# SBATCH --exclude node-2080ti-2
# SBATCH --exclude node-3090-3
# SBATCH --exclude node-1080ti-[0-2],node-2080ti-5
# SBATCH --array=0-180
# SBATCH --array=0-540
# SBATCH --array=0-5
# SBATCH --exclude node-3090-[0-3]
# SBATCH --exclude node-1080ti-[0-2],node-2080ti-0,node-v100-0,node-2080ti-3
# SBATCH --exclude node-3090-0
# SBATCH --nodelist node-3090-[0-2]
# SBATCH --nodelist node-v100-0
# SBATCH --reservation=kostas-cvpr-2022
# SBATCH --nodelist node-2080ti-4

source ~/.bashrc

# Activate the virtual environment
#conda activate pred3d
conda activate turb

NUM_SIMS=30

for VARIABLE in $(seq 1 1 $NUM_SIMS)
do
    python gen_sim.py sim 1
done

