#!/bin/bash
# **The recommended finetuning configurations is to boost your batch size to the max, and train for 20k steps.**

env

echo $NODE_RANK


master_addr=$(scontrol show hostname ${node_list} | head -n1)
# 
echo $master_addr

#bash kill.sh

MASTER_PORT=12111 PYTHONPATH=./ torchrun \
   --nnodes=$SLURM_NNODES \
   --nproc_per_node=1 --master_port=12322 \
   --node_rank=$SLURM_PROCID --master_addr=$master_addr \
   scripts/train/train.py --config run_configs/train/gr00t_n1_sweep.yaml
