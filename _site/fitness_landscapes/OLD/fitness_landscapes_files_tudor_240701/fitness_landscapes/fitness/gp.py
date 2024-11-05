import gpytorch
import numpy as np
import torch

DATA_DIM = 1280

# Source: https://github.com/AIforGreatGood/biotransfer/blob/main/src/models/ExactGP.py
# Will explore, then modify/adapt

class RegressionGP(gpytorch.models.ExactGP):
    """Exact Gaussian Process for regression"""
    def __init__(self, train_x, train_y, likelihood, lengthscale=30):
        """Inits model. See gpytorch for details
        Args:
            train_x: Training dataset input array
            train_y: Training dataset label array
            likelihood: Likehood model from gpytorch
        """
        super(RegressionGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant.data = train_y.mean()
            
        self.covar_module = self.base_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]))
        self.covar_module.base_kernel.lengthscale = lengthscale
        
        v = train_y.var().item()
        likelihood.noise = v/2
        self.covar_module.outputscale = v/2

    def forward(self, x):
        """Forward hook
        Args:
            x: Inference samples
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
###

class SparseGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, likelihood, lengthscale=30, inducing_points=500):
        super(SparseGP, self).__init__(x, y, likelihood)
        self.mean = gpytorch.means.ConstantMean()
        self.mean.constant.data = y.mean()

        self.covar_module = self.base_covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=x.shape[1]))
        self.covar_module.base_kernel.lengthscale = lengthscale

        v = y.var()
        likelihood.noise = v/2
        self.base_covar.outputscale = v/2

        random = np.random.RandomState(1)
        select = random.permutation(len(x))[:inducing_points]
        points = x[select]
        self.covar = gpytorch.kernels.InducingPointKernel(self.base_covar, inducing_points=points, likelihood=likelihood)

    def forward(self, x):
        mu = self.mean(x)
        cov = self.covar(x)
        return gpytorch.distributions.MultivariateNormal(mu, cov)
###

class ExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPHighDim(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPHighDim, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        _, n_dimensions = train_x.shape

        
        mu_0 = 0.0
        sigma_0 = 1.0
        
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                ard_num_dims=n_dimensions,
                lengthscale_prior=gpytorch.priors.LogNormalPrior(
                    mu_0 + np.log(n_dimensions) / 2, sigma_0
                ),
            )
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
###    
class MultitaskGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super(MultitaskGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
