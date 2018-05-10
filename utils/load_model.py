import json
import torch
from jointvae.models import VAE
from utils.dataloaders import (get_mnist_dataloaders, get_dsprites_dataloader,
                               get_chairs_dataloader, get_fashion_mnist_dataloaders)


def load(path):
    """
    Loads a trained model.

    Parameters
    ----------
    path : string
        Path to folder where model is saved. For example
        './trained_models/mnist/'. Note the path MUST end with a '/'
    """
    path_to_specs = path + 'specs.json'
    path_to_model = path + 'model.pt'

    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    # Unpack specs
    dataset = specs["dataset"]
    latent_spec = specs["latent_spec"]

    # Get image size
    if dataset == 'mnist' or dataset == 'fashion_mnist':
        img_size = (1, 32, 32)
    if dataset == 'chairs' or dataset == 'dsprites':
        img_size = (1, 64, 64)
    if dataset == 'celeba':
        img_size = (3, 64, 64)

    # Get model
    model = VAE(img_size=img_size, latent_spec=latent_spec)
    model.load_state_dict(torch.load(path_to_model,
                                     map_location=lambda storage, loc: storage))

    return model
