# Learning Disentangled Joint Continuous and Discrete Representations

Pytorch implementation of [Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/abs/1804.00104) (NIPS 2018).

This repo contains an implementation of JointVAE, a framework for jointly disentangling continuous and discrete factors of variation in data in an unsupervised manner.


## Examples

#### MNIST
<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/mnist_disentangled.gif" width="400">

#### CelebA

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/celeba_disentangled.gif" width="400">

#### FashionMNIST

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/fashion_disentangled.gif" width="400">

#### dSprites

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/dsprites_sweeps.gif" width="400">

#### Discrete and continuous factors on MNIST

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/discrete_continuous_factors.png" width="400">

#### dSprites comparisons

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/dsprites-comparison.png" width="400">

## Usage

The `train_model.ipynb` notebook contains code for training a JointVAE model.

The `load_model.ipynb` notebook contains code for loading a trained model.

#### Example usage
```python
from jointvae.models import VAE
from jointvae.training import Trainer
from torch.optim import Adam
from viz.visualize import Visualizer as Viz

# Build a dataloader for your data
dataloader = get_my_dataloader(batch_size=32)

# Define latent distribution
latent_spec = {'cont': 20, 'disc': [10, 5, 5, 2]}

# Build a Joint-VAE model
model = VAE(img_size=(3, 64, 64), latent_spec=latent_spec)

# Build a trainer and train model
optimizer = Adam(model.parameters())
trainer = Trainer(model, optimizer,
                  cont_capacity=[0., 5., 25000, 30.],
                  disc_capacity=[0., 5., 25000, 30.])
trainer.train(dataloader, epochs=10)

# Visualize samples from the model
viz = Viz(model)
samples = viz.samples()

# Do all sorts of fun things with model
...
```

## Trained models

The trained models referenced in the paper are included in the `trained_models` folder. The `load_model.ipynb` ipython notebook provides code to load and use these trained models.

## Data sources

The MNIST and FashionMNIST datasets can be automatically downloaded using `torchvision`.

#### CelebA
All CelebA images were resized to be 64 by 64. Data can be found [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

#### Chairs
All Chairs images were center cropped and resized to 64 by 64. Data can be found [here](https://www.di.ens.fr/willow/research/seeing3Dchairs/).

## Applications

### Image editing

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/face-manipulation.jpg" width="600">

### Inferring unlabelled quantities

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/inferred-rotation.jpg" width="600">

## Citing

If you find this work useful in your research, please cite using:

```
@inproceedings{dupont2018learning,
  title={Learning disentangled joint continuous and discrete representations},
  author={Dupont, Emilien},
  booktitle={Advances in Neural Information Processing Systems},
  pages={707--717},
  year={2018}
}
```

## More examples

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/mnist_angle.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/mnist_stroke_thickness.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/mnist_width.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/mnist_digit_type_thickness.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/celeba_azimuth_large.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/celeba_background_large.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/celeba_age_large.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/chair_azimuth.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/chair_size.png" width="300">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/chairs_disentangled2.gif" width="200">

<img src="https://github.com/Schlumberger/joint-vae/raw/master/imgs/fashion_traversals.png" width="300">

## License

MIT
