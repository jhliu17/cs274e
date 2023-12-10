#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python reconstruction/inference.py --config="reconstruction/config_vae.json"  --data_path="dataset/shapenet_chair/train.npz" --model_path="outputs/vae_logs_onchair_ep500/epoch_300.pth" --logdir="outputs/inference/vae_logs_onchair_ep500"
