import numpy as np

from . import sd, utils

import torch
from torch import nn
import torch.nn.functional as F


class ExpMapFlow(nn.Module):
    def __init__(self, D):
        super(ExpMapFlow, self).__init__()
        
        self.D = D
        # prepare in advacnce to avoid device moving during forward pass
        self.register_buffer('_Id', torch.eye(self.D + 1), persistent=False)

    def _compute_dF_ddF(self, x):
        raise NotImplementedError("This is an abstract module for inherittance.")

    def _compute_z_J_E(self, x, dF, ddF):
        # project onto the tangent space
        Id = self._Id
        proj = Id - torch.einsum('ni,nj->nij', x, x)
        v = torch.einsum('nij,nj->ni', proj, dF)

        # exponential map transform
        norm_v = torch.norm(v, dim=1, keepdim=True)
        v_unit = v / norm_v
        cos_norm_v = torch.cos(norm_v)
        sin_norm_v = torch.sin(norm_v)
        z = x * cos_norm_v + v_unit * sin_norm_v
        
        # Euclidena Jacobian of the projected gradient
        dv = torch.einsum('nij,njk->nik', proj, ddF) \
                - torch.einsum('ni,nj->nij', x, dF) \
                - torch.sum(x * dF, dim=1)[:, None, None] * Id
   
        # Jacobian of exponential map
        sin_div_norm_v = sin_norm_v / norm_v
        vvT = torch.einsum('ni,nj->nij', v_unit, v_unit)
        xvT = torch.einsum('ni,nj->nij', x, v)
        J = cos_norm_v[..., None] * Id[None, ...]
        J += sin_div_norm_v[..., None] * dv
        J += torch.einsum(
            'nij,njk->nik',
            + cos_norm_v[..., None] * vvT
            - sin_div_norm_v[..., None] * vvT
            - sin_div_norm_v[..., None] * xvT,
            dv
        )

        # orthonomral basis of the tangent space
        E = torch.dstack([
            v_unit,
            torch.cross(x, v_unit)
        ])

        return z, J, E
    
    def forward(self, x):
        dF, ddF = self._compute_dF_ddF(x)

        z, J, E = self._compute_z_J_E(x, dF, ddF)

        JE = torch.matmul(J, E)
        JETJE = torch.einsum('nij,nik->njk', JE, JE)

        return z, 0.5 * torch.slogdet(JETJE)[1]

class ExpMapPolyFlow(ExpMapFlow):
    def __init__(self, D):
        super(ExpMapPolyFlow, self).__init__(D)
        n = (D + 2) * (D + 1)
        self.mu = nn.Parameter(torch.ones(D + 1) / n)
        self.A = nn.Parameter(torch.ones(D + 1, D + 1) / n)

    def _compute_dF_ddF(self, x):
        # parameters of polynormal uTx + xTAx
        mu, A = self.mu, self.A
        # |mu|_1 + |A|_1 <= 1 (|.|_1 is the elementwise l1 norm)
        sum_l1_norm = torch.abs(mu).sum() + torch.abs(A).sum()
        if sum_l1_norm > 1:
            mu = mu / sum_l1_norm
            A = A / sum_l1_norm

        # Euclidean gradient and Jacobian of the scalar field
        dF = mu + 2 * torch.einsum('ij,nj->ni', A, x)
        ddF = (2 * A[None, ...]).repeat(x.size(0), 1, 1)

        return dF, ddF

class ExpMapSumRadFlow(ExpMapFlow):
    def __init__(self, D, K):
        super(ExpMapSumRadFlow, self).__init__(D)

        self.K = K
        # self.mu = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.mu = nn.Parameter(torch.Tensor(K, D+1), requires_grad=True)
        self.beta = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.alpha = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self._initialize_parameters()

    def _initialize_parameters(self):
        dirs, _ = sd.sample_sd(self.D, self.K)
        # self.mu.data.copy_(utils.euclidean_to_spherical(dirs))
        self.mu.data.copy_(dirs)
        self.beta.data.copy_(utils.softplus_inv(
            5 * torch.rand(self.K) + 1
        ))
        self.alpha.data.copy_(utils.softplus_inv(
            torch.ones(self.K) / self.K
        ))

    def _compute_dF_ddF(self, x):
        # parameters of radial components
        mu = self.mu / torch.norm(self.mu, dim=1, keepdim=True)
        # mu = utils.spherical_to_euclidean(self.mu)
        beta = F.softplus(self.beta)
        
        # alpha_i >= 0 and \sum_i alpha_i <= 1
        alpha = F.softplus(self.alpha)
        if alpha.sum() > 1:
            alpha = alpha / alpha.sum()

        # Euclidean gradient and Jacobian of the scalar field
        exp_factor = torch.exp(beta * (torch.matmul(x, mu.T) - 1))
        dF = torch.matmul(exp_factor * alpha, mu)
        ddF = torch.einsum(
            'ndk,pk->npd',
            torch.einsum('nk,dk->ndk', exp_factor, alpha * beta * mu.T),
            mu.T
        )
        return dF, ddF

class EMP(nn.Module):
    def __init__(self, D, N):
        super(EMP, self).__init__()
        
        self.D = D
        self.N = N
        layers = []
        for i in range(N):
            layers.append(
                ExpMapPolyFlow(D)
            )        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.atleast_2d(x)
        assert x.ndim == 2 and x.size(1) == self.D + 1

        ldjs = 0
        for layer in self.layers:
            x, ldj = layer(x)
            ldjs += ldj
        return x, ldjs

class EMSRE(nn.Module):
    def __init__(self, D, N, K):
        super(EMSRE, self).__init__()
        
        self.D = D
        self.N = N
        self.K = K
        layers = []
        for i in range(N):
            layers.append(
                ExpMapSumRadFlow(D, K)
            )        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x = torch.atleast_2d(x)
        assert x.ndim == 2 and x.size(1) == self.D + 1

        ldjs = 0
        for layer in self.layers:
            x, ldj = layer(x)
            ldjs += ldj
        return x, ldjs

# ReLU-MLP [Ni, 64, 64, No]
class ConditionalTransform(nn.Module):
    def __init__(self, Ni, No):
        super(ConditionalTransform, self).__init__()

        layers = []
        layers.append(nn.Linear(Ni, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 64))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, No))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class CircularSplineTransform(nn.Module):
    def __init__(self, K, n_cond=0, lb=(-1, -1), rt=(1, 1)):
        super(CircularSplineTransform, self).__init__()

        assert n_cond >= 0
        assert rt[0] > lb[0] and rt[1] > lb[1]

        self.K = K
        self.n_cond = n_cond
        self.lb = lb
        self.rt = rt

        if n_cond == 0:
            self.theta_w = nn.Parameter(torch.randn(1, K))
            self.theta_h = nn.Parameter(torch.randn(1, K))
            self.theta_d = nn.Parameter(utils.softplus_inv(torch.ones(1, K-1)))
        else:
            self.conditioner = ConditionalTransform(n_cond, 3 * K - 1)

    def forward(self, x, z=None):
        if self.n_cond == 0:
            theta_w = self.theta_w
            theta_h = self.theta_h
            theta_d = self.theta_d
        else:
            assert z!=None, "This module is conditional on preceeding dimensions!"
            theta = self.conditioner(z)
            theta_w, theta_h, theta_d = torch.split(theta, [self.K, self.K, self.K-1], dim=1)

        width = torch.softmax(theta_w, dim=1) * (self.rt[0] - self.lb[0])
        height = torch.softmax(theta_h, dim=1) * (self.rt[1] - self.lb[1])

        x_ks = F.pad(torch.cumsum(width, dim=1), (1, 0), mode='constant', value=0.) + self.lb[0]
        y_ks = F.pad(torch.cumsum(height, dim=1), (1, 0), mode='constant', value=0.) + self.lb[1]

        derivatives = F.pad(torch.nn.Softplus()(theta_d) + 1e-3, (1, 1), mode='constant',
                        value=(self.rt[1] - self.lb[1]) / (self.rt[0] - self.lb[0]))

        return self._forward(x, x_ks, y_ks, derivatives)

    def _forward(self, x, x_ks, y_ks, derivatives):
        k = torch.searchsorted(x_ks.squeeze(dim=0), x).squeeze(dim=1)

        k = torch.where(k == 0, 1, k)
        k = torch.where(k == self.K + 1, self.K, k)
        k = k - 1

        nk = k + 1 # next k

        if self.n_cond == 0:
            x_k = x_ks[0, k]
            x_nk = x_ks[0, nk]
            y_k = y_ks[0, k]
            y_nk = y_ks[0, nk]
            d_k = derivatives[0, k]
            d_nk = derivatives[0, nk]
        else:
            bs = len(x)
            batch_ind = torch.arange(bs).to(x.device)
            x_k = x_ks[batch_ind, k]
            x_nk = x_ks[batch_ind, nk]
            y_k = y_ks[batch_ind, k]
            y_nk = y_ks[batch_ind, nk]
            d_k = derivatives[batch_ind, k]
            d_nk = derivatives[batch_ind, nk]

        s_k = (y_nk - y_k) / (x_nk - x_k)
        eps = (x.squeeze(dim=1) - x_k) / (x_nk - x_k)

        # transformed x
        tx = y_k + (y_nk - y_k) * (s_k * (eps ** 2) + d_k * eps * (1 - eps)) / \
                        (s_k + (d_nk + d_k - 2 * s_k) * eps * (1 - eps))

        dtx = (s_k ** 2) * (d_nk * (eps ** 2) + 2 * s_k * eps * (1 - eps) + d_k * ((1 - eps) ** 2)) / \
                        ((s_k + (d_nk + d_k - 2 * s_k) * eps * (1 - eps)) ** 2)

        return tx[..., None], dtx


class MobiusTransform(nn.Module):
    def __init__(self, K, n_cond=0):
        super(MobiusTransform, self).__init__()

        assert n_cond >= 0
        self.K = K
        self.n_cond = n_cond

        if n_cond == 0:
            self.weights = nn.Parameter(torch.randn(1, K))
            self.w = nn.Parameter(torch.randn(1, K, 2))
        else:
            self.conditioner = ConditionalTransform(n_cond, 3 * K)

        self.register_buffer('_zero_radian', torch.tensor([[1, 0]]), persistent=False)
        self.register_buffer('_I', torch.eye(2), persistent=False)


    def forward(self, x, z=None):
        if self.n_cond == 0:
            weights = self.weights
            w = self.w
        else:
            assert z!=None, "This module is conditional on preceeding dimensions!"
            conditions = self.conditioner(z)
            weights, w = torch.split(conditions, [self.K, 2 * self.K], dim=1)
            w = w.reshape(-1, self.K, 2)

        weights = torch.softmax(weights, dim=1)
        w = 0.99 / (1 + torch.norm(w, dim=-1, keepdim=True)) * w

        return self._forward(x, weights, w)

    def _h(self, z, w):
        w_norm = torch.norm(w, dim=-1, keepdim=True) # n x k x 1

        h_z = (1 - w_norm ** 2) / (torch.norm((z.reshape(-1, 1, 2) - w), dim=-1, keepdim=True) ** 2)* \
                (z.reshape(-1, 1, 2) - w) - w
        
        return h_z


    def _forward(self, x, weights, w):
        # weights - n x k
        # w - n x k x 2
        z = torch.hstack([
            torch.cos(x), torch.sin(x)
        ]) # n x 2

        h_z = self._h(z, w)
        h_zero_radian = self._h(self._zero_radian, w)

        radians = torch.atan2(h_z[..., 1], h_z[..., 0])
        shifts = torch.atan2(h_zero_radian[..., 1], h_zero_radian[..., 0])

        tx = radians - shifts 
        tx = torch.where(tx >= 0, tx, tx + torch.pi * 2)
        tx = torch.sum(weights * tx, dim=1, keepdim=True)


        z_w = z[:, None, :] - w
        z_w_norm = torch.norm(z_w, dim=-1)
        z_w_unit = z_w / z_w_norm[..., None]
        
        # n x 2
        dz_dtheta = torch.hstack([
            -torch.sin(x), torch.cos(x)
        ]) 

        # n x k x 2 x 2
        dh_dz = (1 - torch.norm(w, dim=-1) ** 2)[..., None, None] * \
                    (self._I[None, None, ...] - 2 * torch.einsum('nki,nkj->nkij', z_w_unit, z_w_unit)) / \
                        (z_w_norm[..., None, None] ** 2)
        
        dh_dtheta = torch.einsum('nkpq,nq->nkp',dh_dz, dz_dtheta)
        dtx = torch.sum(torch.norm(dh_dtheta, dim=-1) * weights, dim=1)
        return tx, dtx

class MobiusSplineFlow(nn.Module):
    def __init__(self, D, Km, Ks, ordering, hemi=False):
        super(MobiusSplineFlow, self).__init__()
        # TODO: generalize D
        if D != 2:
            raise NotImplementedError("only the case D=2 is implemented")

        self.D = D
        self.Km = Km
        self.Ks = Ks
        self.ordering = ordering
        self.hemi = hemi
        
        if hemi:
            lb, rt = (0, 0), (1, 1)
        else:
            lb, rt = (-1, -1), (1, 1)

        if ordering == 0:
            self.spline = CircularSplineTransform(Ks, lb=lb, rt=rt)
            self.mobius = MobiusTransform(Km, n_cond=1)
        elif ordering == 1:
            self.spline = CircularSplineTransform(Ks, n_cond=1, lb=lb, rt=rt)
            self.mobius = MobiusTransform(Km)
        else:
            raise NotImplementedError("only the case D=2 is implemented, so only 1 or 0 ordering")
        

    def forward(self, r, z):

        if self.ordering == 0:
            tr, dtr = self.spline(r)
            tz, dtz = self.mobius(z, r)
        elif self.ordering == 1:
            tz, dtz = self.mobius(z)
            tr, dtr = self.spline(r, z)

        ldj = torch.log(dtr) + torch.log(dtz)

        return tr, tz, ldj

class MS(nn.Module):
    def __init__(self, D, N, Km, Ks, hemi=False):
        super(MS, self).__init__()

        # TODO: generalize D
        if D != 2:
            raise NotImplementedError("only the case D=2 is implemented")

        self.D = D
        self.N = N
        self.Km = Km
        self.Ks = Ks
        self.hemi = hemi
        
        layers = []
        ordering = 0
        for _ in range(N):
            layer = MobiusSplineFlow(D, Km, Ks, ordering, hemi)
            layers.append(layer)
            ordering = 1 - ordering
        
        self.layers = nn.ModuleList(layers)

    def forward(self, theta, phi):
        theta = torch.atleast_2d(theta)
        phi = torch.atleast_2d(phi)
        
        r = torch.cos(theta)
        z = phi

        ldjs = 0
        for layer in self.layers:
            r, z, ldj = layer(r, z)
            ldjs += ldj

        ttheta = torch.acos(r)
        tphi = z
        
        return ttheta, tphi, ldjs


        


