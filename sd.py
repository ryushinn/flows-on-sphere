import torch

def uniform_sample_s2(num_samples):
    r"""Uniformly sample points on \mathbb{S}^2."""
    phi = 2 * torch.pi * torch.rand(num_samples);
    theta = torch.acos(1 - 2 * torch.rand(num_samples));
    phi = phi[..., None]
    theta = theta[..., None]

    log_prob = torch.log((1 / (4 * torch.pi)) * torch.ones(num_samples))

    return torch.concat((
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ), dim=1), log_prob

def sample_sd(D, num_samples, method='uniform'):
    if (D != 2 or method != 'uniform'):
        raise NotImplementedError(f"The {method} sampling on S{D} space is not implemented")
    return uniform_sample_s2(num_samples)


def expmap(x, v):
    r"""Exponential map on \mathbb{S}^D.
    Args:
        x: points on \mathbb{S}^D, embedded in \mathbb{R}^{D+1}
        v: vectors in the tangent space T_x \mathbb{S}^D
    Returns:
        Image of exponential map
    """
    v_norm = torch.norm(v, dim=1, keepdim=True)
    return x * torch.cos(v_norm) + (v / v_norm) * torch.sin(v_norm)

