import torch
import numpy as np

# coordinate conversions
def spherical_to_euclidean(sph_coords):
    sph_coords = torch.atleast_2d(sph_coords)
    assert sph_coords.ndim ==2 and sph_coords.size(1) == 2

    theta, phi = torch.split(sph_coords, (1, 1), 1)
    return torch.concat((
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ), dim=1)
def euclidean_to_spherical(euc_coords):
    euc_coords = torch.atleast_2d(euc_coords)
    assert euc_coords.ndim == 2 and euc_coords.size(1) == 3

    x, y, z = torch.split(euc_coords, (1, 1, 1), 1)
    
    return torch.concat((
        torch.arccos(z),
        torch.pi + torch.arctan2(-y, -x),
    ), dim=1)

# some general functions
def softplus(x):
    return torch.log(1. + torch.exp(x))
def softplus_inv(x):
    return torch.log(-1. + torch.exp(x))
def softmax(x):
    ex = torch.exp(x - torch.max(x))
    return ex / torch.sum(ex)

# target density used in the paper
_target_mu = spherical_to_euclidean(torch.Tensor([
    [0.7 + np.pi / 2, 1.5],
    [-1. + np.pi / 2, 1.],
    [0.6 + np.pi / 2, 5],  # 0.5 -> 5.!
    [-0.7 + np.pi / 2, 4.]
]))
def s2_target(x, device='cpu'):
    xe = torch.matmul(x, _target_mu.T.to(device))
    return torch.sum(torch.exp(10 * xe), dim=1)

# metrics reported in the paper
def kl_ess(log_model_prob, target_prob):
    weights = target_prob / torch.exp(log_model_prob)
    Z = torch.mean(weights)
    KL = torch.mean(log_model_prob - torch.log(target_prob)) + torch.log(Z)
    ESS = torch.sum(weights) ** 2 / torch.sum(weights ** 2)
    return Z, KL, ESS