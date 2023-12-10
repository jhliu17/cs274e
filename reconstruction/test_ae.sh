#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python reconstruction/reconstruction_test.py --config="reconstruction/config.json" --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" --model_path="outputs/ae_emd_logs_ep500/epoch_300.pth" --logdir="outputs/reconstruction/ae_emd_logs_ep500"
