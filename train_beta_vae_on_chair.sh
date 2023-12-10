export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python train.py --config="config_vae.json" --logdir="outputs/beta_vae_onchair_logs_ep500/" --data_path="dataset/shapenet_chair/train.npz" --loss="swd" --autoencoder="pointnet_vae" --beta_vae_beta=5
