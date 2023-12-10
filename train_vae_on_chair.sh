export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python train.py --config="config_vae.json" --logdir="outputs/vae_logs_onchair_ep1000/" --data_path="dataset/shapenet_chair/train.npz" --loss="swd" --autoencoder="pointnet_vae"
