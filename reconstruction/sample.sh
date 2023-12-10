#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python reconstruction/sampling.py --config="reconstruction/config_vae.json"  --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" --model_path="outputs/vae_logs_onchair_ep500/epoch_1000.pth" --logdir="outputs/sample/vae_swd_logs_onchair"
