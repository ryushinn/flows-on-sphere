import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch

import utils

NUM_POINTS = 200

theta = np.linspace(0, np.pi, NUM_POINTS)
phi = np.linspace(0, 2 * np.pi, NUM_POINTS)
_tp = np.array(np.meshgrid(theta, phi, indexing='ij'), dtype=np.float32)
_tp = _tp.transpose([1, 2, 0]).reshape(-1, 2)
_tp = torch.from_numpy(_tp)

def _to_numpy(x):
    return x.cpu().numpy()

def plot_model_density(model_samples, save=False, path=None):
    estimated_density = gaussian_kde(
        _to_numpy(utils.euclidean_to_spherical(model_samples).T), 0.2)
    heatmap = estimated_density(_tp.T).reshape(NUM_POINTS, NUM_POINTS)
    _plot_mollweide(heatmap, save, path)


def plot_target_density(target_fn, save=False, path=None):
    density = target_fn(utils.spherical_to_euclidean(_tp))
    heatmap = density.reshape(NUM_POINTS, NUM_POINTS)
    _plot_mollweide(_to_numpy(heatmap), save, path)


def _plot_mollweide(heatmap, save=False, path=None):
    tt, pp = np.meshgrid(theta - np.pi / 2, phi - np.pi, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.pcolormesh(pp, tt, heatmap, cmap=plt.cm.jet)
    ax.set_axis_off()
    if save and path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()