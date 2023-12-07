#!/bin/sh
# python reconstruction/reconstruction_test.py --config="reconstruction/config.json" --logdir="logs/" --data_path="dataset/modelnet40_ply_hdf5_2048/"
# python reconstruction/inference.py --config="reconstruction/config.json" --logdir="logs/" --data_path="dataset/modelnet40_ply_hdf5_2048/"
python reconstruction/inference.py --config="reconstruction/config_shapenet55.json" --logdir="logs/" --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz"
