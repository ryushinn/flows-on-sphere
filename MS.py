import argparse
import torch
from torch import optim
import numpy as np
from tqdm import tqdm
import sd, utils, flows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fit the target density (specified in the paper) with Mobius Spline flows.')
    parser.add_argument('--N', type=int, default=1, help='the number of stacked flows')
    parser.add_argument('--Km', type=int, default=6, help='the number of centers of mobius transforms')
    parser.add_argument('--Ks', type=int, default=12, help='the number of segments of spline transforms')
    parser.add_argument('--n_iter', type=int, default=20000, help='the number of iterations')
    parser.add_argument('--n_disp_iter', type=int, default=2000, help='the number of iteration interval of displaying validating results')
    parser.add_argument('--lr', type=float, default=2e-4, help='the learning rate for training flows')
    parser.add_argument('--bs', type=int, default=256, help='the batch size for training flows')
    parser.add_argument('--n_val_samples', type=int, default=20000, help='the number of validating samples')
    parser.add_argument('--RD_SEED', type=int, default=42, help='the random seed used for reproducible experiments')
    args = parser.parse_args()

    D = 2
    N = args['N']
    Km = args['Km']
    Ks = args['Ks']

    n_iter = args['n_iter']
    display_iter = args['n_disp_iter']

    lr = args['lr']
    bs = args['bs']
    n_val_samples = args['n_val_samples']

    # for reproducible experiments
    RD_SEED = args['RD_SEED']
    torch.manual_seed(RD_SEED);  # reproducibility
    np.random.seed(RD_SEED)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    MS = flows.MS(D, N, Km, Ks)
    MS = MS.to(device)

    optimizer = optim.Adam(MS.parameters(), lr=lr)

    sampler = sd.batch_sampler(D, bs)
    val_samples, val_log_prob = sd.sample_sd(D, n_val_samples)
    val_samples, val_log_prob = val_samples.to(device), val_log_prob.to(device)

    MS.train()
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

            z, ldjs = MS(x)

            loss = torch.mean(log_prob - ldjs - torch.log(s2_target(z, device)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ((iter + 1) % display_iter == 0):
                with torch.no_grad():
                    z, ldjs = MS(val_samples)
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