#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python reconstruction/inference.py --config="reconstruction/config_vae.json" --logdir="outputs/inference/logs/" --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz"
