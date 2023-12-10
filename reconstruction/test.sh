#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python reconstruction/reconstruction_test.py --config="reconstruction/config_vae.json" --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" --model_path="outputs/beta_vae_chamfer_logs_ep500/epoch_800.pth" --logdir="outputs/reconstruction/beta_vae_chamfer_logs_ep500"
