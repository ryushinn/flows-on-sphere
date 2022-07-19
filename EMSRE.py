import argparse
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from . import sd, utils, flows, visualize


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fit the target density (specified in the paper) with Mobius Spline flows.')
    parser.add_argument('--N', type=int, default=1, help='the number of stacked flows')
    parser.add_argument('--K', type=int, default=6, help='the number of radial components in the scalar field')
    parser.add_argument('--n_iter', type=int, default=20000, help='the number of iterations')
    parser.add_argument('--n_disp_iter', type=int, default=2000, help='the number of iteration interval of displaying validating results')
    parser.add_argument('--lr', type=float, default=2e-4, help='the learning rate for training flows')
    parser.add_argument('--bs', type=int, default=256, help='the batch size for training flows')
    parser.add_argument('--n_val_samples', type=int, default=20000, help='the number of validating samples')
    parser.add_argument('--RD_SEED', type=int, default=42, help='the random seed used for reproducible experiments')
    parser.add_argument('--save_path', type=str, default='.', help='the path of saving results')
    args = parser.parse_args()


    # hyperparameters
    D = 2
    N = args.N
    K = args.K

    n_iter = args.n_iter
    display_iter = args.n_disp_iter

    lr = args.lr
    bs = args.bs
    n_val_samples = args.n_val_samples
    
    path = args.save_path

    # for reproducible experiments
    RD_SEED = args.RD_SEED
    torch.manual_seed(RD_SEED);  # reproducibility
    np.random.seed(RD_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'


    # training
    EMSRE = flows.EMSRE(D, N, K)
    EMSRE = EMSRE.to(device)

    optimizer = optim.Adam(EMSRE.parameters(), lr=lr)

    sampler = sd.batch_sampler(D, bs)
    val_samples, val_log_prob = sd.sample_sd(D, n_val_samples)
    val_samples, val_log_prob = val_samples.to(device), val_log_prob.to(device)

    EMSRE.train()
    losses = []
    val_kl = []
    val_ess = []
    with tqdm(total=n_iter) as t:
        val_KL = 'N/A'
        val_ESS = 'N/A'
        for iter in range(n_iter):
            logs = {}
            x, log_prob = next(sampler)
            x = x.to(device)
            log_prob = log_prob.to(device)

            z, ldjs = EMSRE(x)

            loss = torch.mean(log_prob - ldjs - torch.log(utils.s2_target(z, device)))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((iter + 1) % display_iter == 0):
                with torch.no_grad():
                    z, ldjs = EMSRE(val_samples)
                    target_prob = utils.s2_target(z, device)
                    _, KL, ESS = utils.kl_ess(val_log_prob - ldjs, target_prob)
                    val_kl.append(KL.item())
                    val_ess.append(ESS.item() / n_val_samples * 100)
                    val_KL = f'{KL.item():.3f}'
                    val_ESS = f'{ESS.item() / n_val_samples * 100:.0f}%'

            losses.append(loss.item())
            logs['loss'] = f'{loss.item():.3f}'
            logs['val_KL'] = val_KL
            logs['val_ESS'] = val_ESS
            t.set_postfix(logs)
            t.update()

    # saving
    fig, axes = plt.subplots(3, 1, figsize=(8, 15))
    axes[0].plot(losses)
    axes[1].plot(val_kl)
    axes[2].plot(val_ess)
    plt.savefig(os.path.join(path, 'metrics.png'), bbox_inches='tight')
    plt.show()

    vis_samples, _ = sd.sample_sd(D, 2500)
    with torch.no_grad():
        vis_samples = vis_samples.to(device)
        z, ldjs = EMSRE(vis_samples)
        visualize.plot_model_density(z, save=True, path=os.path.join(path, 'flow_density.png'))