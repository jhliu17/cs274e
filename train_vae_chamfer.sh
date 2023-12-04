export CUDA_VISIBLE_DEVICES=$1

module load cuda/11.3
python train.py --config="config_vae.json" --logdir="outputs/vae_chamfer_logs_ep500/" --data_path="dataset/shapenet_core55/shapenet57448xyzonly.npz" --loss="chamfer" --autoencoder="pointnet_vae"
