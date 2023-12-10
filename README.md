# CS 274E Deep Generative Model Class Project

Project title: **Accurate and Efficient Representation Learning of 3D Point Cloud Data with VAE**

Group members: Thanh-Tung Le, Junhao Liu, Pooya Khosrav


## Preparation

Please refer to [PointSWD](https://github.com/VinAIResearch/PointSWD/tree/main) for the details of dataset downloading, CUDA kernel compilation, and environment settings.


## Training

Autoencoder

```bash
sh train_ae.sh 0
```

Variational Autoencoder

```bash
# SWD loss
sh train_vae.sh 0

# EMD loss
sh train_vae_emd.sh 0

# Chamfer loss
sh train_vae_chamfer.sh 0
```

$\beta$-Variational Autoencoder: change `--beta_vae_beta` to control $\beta$ value

```bash
# SWD loss
sh train_beta_vae.sh 0

# EMD loss
sh train_beta_vae_emd.sh 0

# Chamfer loss
sh train_beta_vae_chamfer.sh 0
```

## Acknowledgements

Most of code borrowed from [PointSWD](https://github.com/VinAIResearch/PointSWD/tree/main)
