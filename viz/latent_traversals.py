import numpy as np
import torch
from scipy import stats


class LatentTraverser():
    def __init__(self, latent_spec):
        """
        LatentTraverser is used to generate traversals of the latent space.

        Parameters
        ----------
        latent_spec : dict
            See jointvae.models.VAE for parameter definition.
        """
        self.latent_spec = latent_spec
        self.sample_prior = False  # If False fixes samples in untraversed
                                   # latent dimensions. If True samples
                                   # untraversed latent dimensions from prior.
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.cont_dim = latent_spec['cont'] if self.is_continuous else None
        self.disc_dims = latent_spec['disc'] if self.is_discrete else None

    def traverse_line(self, cont_idx=None, disc_idx=None, size=5):
        """
        Returns a (size, D) latent sample, corresponding to a traversal of the
        latent variable indicated by cont_idx or disc_idx.

        Parameters
        ----------
        cont_idx : int or None
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and cont_idx = 7, then the 7th dimension
            will be traversed while all others will either be fixed or randomly
            sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        disc_idx : int or None
            Index of discrete latent dimension to traverse. If there are 5
            discrete latent variables and disc_idx = 3, then only the 3rd
            discrete latent will be traversed while others will be fixed or
            randomly sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        size : int
            Number of samples to generate.
        """
        samples = []

        if self.is_continuous:
            samples.append(self._traverse_continuous_line(idx=cont_idx,
                                                          size=size))
        if self.is_discrete:
            for i, disc_dim in enumerate(self.disc_dims):
                if i == disc_idx:
                    samples.append(self._traverse_discrete_line(dim=disc_dim,
                                                                traverse=True,
                                                                size=size))
                else:
                    samples.append(self._traverse_discrete_line(dim=disc_dim,
                                                                traverse=False,
                                                                size=size))

        return torch.cat(samples, dim=1)

    def _traverse_continuous_line(self, idx, size):
        """
        Returns a (size, cont_dim) latent sample, corresponding to a traversal
        of a continuous latent variable indicated by idx.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        size : int
            Number of samples to generate.
        """
        if self.sample_prior:
            samples = np.random.normal(size=(size, self.cont_dim))
        else:
            samples = np.zeros(shape=(size, self.cont_dim))

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            cdf_traversal = np.linspace(0.05, 0.95, size)
            cont_traversal = stats.norm.ppf(cdf_traversal)

            for i in range(size):
                samples[i, idx] = cont_traversal[i]

        return torch.Tensor(samples)

    def _traverse_discrete_line(self, dim, traverse, size):
        """
        Returns a (size, dim) latent sample, corresponding to a traversal of a
        discrete latent variable.

        Parameters
        ----------
        dim : int
            Number of categories of discrete latent variable.

        traverse : bool
            If True, traverse the categorical variable otherwise keep it fixed
            or randomly sample.

        size : int
            Number of samples to generate.
        """
        samples = np.zeros((size, dim))

        if traverse:
            for i in range(size):
                samples[i, i % dim] = 1.
        else:
            # Randomly select discrete variable (i.e. sample from uniform prior)
            if self.sample_prior:
                samples[np.arange(size), np.random.randint(0, dim, size)] = 1.
            else:
                samples[:, 0] = 1.

        return torch.Tensor(samples)

    def traverse_grid(self, cont_idx=None, cont_axis=None, disc_idx=None,
                      disc_axis=None, size=(5, 5)):
        """
        Returns a (size[0] * size[1], D) latent sample, corresponding to a
        two dimensional traversal of the latent space.

        Parameters
        ----------
        cont_idx : int or None
            Index of continuous dimension to traverse. If the continuous latent
            vector is 10 dimensional and cont_idx = 7, then the 7th dimension
            will be traversed while all others will either be fixed or randomly
            sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        cont_axis : int or None
            Either 0 for traversal across the rows or 1 for traversal across
            the columns. If None and disc_axis not None will default to axis
            which disc_axis is not. Otherwise will default to 0.

        disc_idx : int or None
            Index of discrete latent dimension to traverse. If there are 5
            discrete latent variables and disc_idx = 3, then only the 3rd
            discrete latent will be traversed while others will be fixed or
            randomly sampled. If None, no latent is traversed and all latent
            dimensions are randomly sampled or kept fixed.

        disc_axis : int or None
            Either 0 for traversal across the rows or 1 for traversal across
            the columns. If None and cont_axis not None will default to axis
            which cont_axis is not. Otherwise will default to 1.

        size : tuple of ints
            Shape of grid to generate. E.g. (6, 4).
        """
        if cont_axis is None and disc_axis is None:
            cont_axis = 0
            disc_axis = 0
        elif cont_axis is None:
            cont_axis = int(not disc_axis)
        elif disc_axis is None:
            disc_axis = int(not cont_axis)

        samples = []

        if self.is_continuous:
            samples.append(self._traverse_continuous_grid(idx=cont_idx,
                                                          axis=cont_axis,
                                                          size=size))
        if self.is_discrete:
            for i, disc_dim in enumerate(self.disc_dims):
                if i == disc_idx:
                    samples.append(self._traverse_discrete_grid(dim=disc_dim,
                                                                axis=disc_axis,
                                                                traverse=True,
                                                                size=size))
                else:
                    samples.append(self._traverse_discrete_grid(dim=disc_dim,
                                                                axis=disc_axis,
                                                                traverse=False,
                                                                size=size))

        return torch.cat(samples, dim=1)

    def _traverse_continuous_grid(self, idx, axis, size):
        """
        Returns a (size[0] * size[1], cont_dim) latent sample, corresponding to
        a two dimensional traversal of the continuous latent space.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        axis : int
            Either 0 for traversal across the rows or 1 for traversal across
            the columns.

        size : tuple of ints
            Shape of grid to generate. E.g. (6, 4).
        """
        num_samples = size[0] * size[1]

        if self.sample_prior:
            samples = np.random.normal(size=(num_samples, self.cont_dim))
        else:
            samples = np.zeros(shape=(num_samples, self.cont_dim))

        if idx is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            cdf_traversal = np.linspace(0.05, 0.95, size[axis])
            cont_traversal = stats.norm.ppf(cdf_traversal)

            for i in range(size[0]):
                for j in range(size[1]):
                    if axis == 0:
                        samples[i * size[1] + j, idx] = cont_traversal[i]
                    else:
                        samples[i * size[1] + j, idx] = cont_traversal[j]

        return torch.Tensor(samples)

    def _traverse_discrete_grid(self, dim, axis, traverse, size):
        """
        Returns a (size[0] * size[1], dim) latent sample, corresponding to a
        two dimensional traversal of a discrete latent variable, where the
        dimension of the traversal is determined by axis.

        Parameters
        ----------
        idx : int or None
            Index of continuous latent dimension to traverse. If None, no
            latent is traversed and all latent dimensions are randomly sampled
            or kept fixed.

        axis : int
            Either 0 for traversal across the rows or 1 for traversal across
            the columns.

        traverse : bool
            If True, traverse the categorical variable otherwise keep it fixed
            or randomly sample.

        size : tuple of ints
            Shape of grid to generate. E.g. (6, 4).
        """
        num_samples = size[0] * size[1]
        samples = np.zeros((num_samples, dim))

        if traverse:
            disc_traversal = [i % dim for i in range(size[axis])]
            for i in range(size[0]):
                for j in range(size[1]):
                    if axis == 0:
                        samples[i * size[1] + j, disc_traversal[i]] = 1.
                    else:
                        samples[i * size[1] + j, disc_traversal[j]] = 1.
        else:
            # Randomly select discrete variable (i.e. sample from uniform prior)
            if self.sample_prior:
                samples[np.arange(num_samples), np.random.randint(0, dim, num_samples)] = 1.
            else:
                samples[:, 0] = 1.

        return torch.Tensor(samples)