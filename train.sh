export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python train.py --config="config.json" --logdir="outputs/logs/" --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" --loss="swd" --autoencoder="pointnet"
