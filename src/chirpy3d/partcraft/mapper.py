from typing import Optional

import torch
from torch import nn, distributions


def create_projection(
    in_dims, out_dims, projection_nlayers, projection_activation, h_dims=0
):
    projections = []
    if h_dims == 0:
        h_dims = out_dims

    for i in range(projection_nlayers - 1):
        projections.append(nn.Linear(in_dims, h_dims))
        projections.append(projection_activation)
        in_dims = h_dims

    projections.append(nn.Linear(in_dims, out_dims))
    return nn.Sequential(*projections)


class TokenMapper(nn.Module):
    def __init__(
        self,
        num_parts,
        num_k_per_part,
        out_dims,
        projection_nlayers=1,
        projection_activation=nn.ReLU(),
        with_pe=True,
        vae_dims=0,
        concat_pe=False,
        learnable_dino_embeddings=False,
        dino_nlayers=2,
        h_dims=0,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.num_k_per_part = num_k_per_part
        self.with_pe = with_pe
        self.out_dims = out_dims

        if vae_dims == 0:
            vae_dims = out_dims

        self.vae_dims = vae_dims
        self.concat_pe = concat_pe
        self.learnable_dino_embeddings = learnable_dino_embeddings

        self.embedding = nn.Embedding((self.num_k_per_part + 1) * num_parts, vae_dims)

        dino_in = 768
        dino_out = num_parts * vae_dims

        self.dino_embeddings = nn.Embedding(self.num_k_per_part + 1, dino_in)
        self.dino_mapper = create_projection(
            dino_in, dino_out, dino_nlayers, projection_activation, dino_in
        )

        # nn.init.zeros_(self.embedding.weight)
        nn.init.normal_(self.embedding.weight.data, 0.0, 0.01)

        if with_pe:
            s = 1.0

            if concat_pe:
                pe_dims = vae_dims
                self.pe = nn.Parameter(torch.randn(num_parts, pe_dims) * s)
            else:
                self.pe = nn.Parameter(torch.randn(num_parts, out_dims) * s)
        else:
            self.register_buffer("pe", torch.zeros(num_parts, out_dims))

        if projection_nlayers == 0:
            assert out_dims == vae_dims
            self.projection = nn.Identity()
        else:
            in_dims = vae_dims

            if self.concat_pe:
                in_dims += vae_dims

            self.projection = create_projection(
                in_dims, out_dims, projection_nlayers, projection_activation, h_dims
            )

    def forward_dino(self, dino_feats, hashes=None, update=False, ema_rate=0.9):
        # dino_feats: (B, C)
        B, C = dino_feats.size()
        class_idxs = hashes.min(dim=1)[0].long()  # because max = num_k_per_part, (B,)

        if self.learnable_dino_embeddings:  # only works in training mode
            assert hashes is not None and self.training
            dino_feats = self.dino_embeddings(class_idxs)  # (B, C)
        elif update:
            # update class-level embeddings EMA
            grad_enabled = torch.is_grad_enabled()

            torch.set_grad_enabled(False)

            for c in class_idxs.unique():
                mask = class_idxs == c
                dino_feats_c = dino_feats[mask]
                # logvar_c = logvar[mask, i]
                dino_feats_c = dino_feats_c.mean(dim=0)  # (D,)

                self.dino_embeddings.weight.data[c] = (
                    ema_rate * self.dino_embeddings.weight.data[c]
                    + (1 - ema_rate) * dino_feats_c
                )

            torch.set_grad_enabled(grad_enabled)

        mu = self.dino_mapper(dino_feats).reshape(B, self.num_parts, -1)
        logvar = None

        if update:
            # update part-level embeddings

            grad_enabled = torch.is_grad_enabled()

            torch.set_grad_enabled(False)

            assert hashes is not None
            cls_idxs = hashes.long()  # (B, N)

            for i in range(self.num_parts):
                unique_cls = cls_idxs[:, i].unique()

                for c in unique_cls:
                    mask = cls_idxs[:, i] == c
                    mu_c = mu[mask, i]
                    # logvar_c = logvar[mask, i]
                    mu_c = mu_c.mean(dim=0)  # (D,)

                    emb_idx = i * (self.num_k_per_part + 1) + c
                    self.embedding.weight.data[emb_idx] = (
                        ema_rate * self.embedding.weight.data[emb_idx]
                        + (1 - ema_rate) * mu_c
                    )

            torch.set_grad_enabled(grad_enabled)

        sampled = mu

        return sampled, mu, logvar

    def obtain_code_mu(self, partid, kid, use_dino_features=False):
        # partid: torch.tensor([0,1,2])
        # kid: torch.tensor([16,16,16])
        # output: torch.tensor(...) N x D

        if use_dino_features and self.learnable_dino_embeddings:
            dino_feats = self.dino_embeddings(kid.long())  # (N, C)
            output = self.dino_mapper(dino_feats).reshape(kid.size(0), -1)

            logvar = None
            mus = []

            for i in range(kid.size(0)):
                mu = output[
                    i, partid[i] * self.vae_dims : (partid[i] + 1) * self.vae_dims
                ]
                mus.append(mu)

            output = torch.stack(mus)
        else:
            output = self.embedding(partid * (self.num_k_per_part + 1) + kid)

        return output

    def random_sample_mu(
        self, batch_size, partid=None, use_dino_features=False, covariance=False
    ):
        # partid: torch.tensor([0,1,2])
        if partid is None:
            partid = torch.arange(
                0, self.num_parts, device=self.embedding.weight.device
            )

        if use_dino_features and self.learnable_dino_embeddings:
            kids = torch.arange(0, self.num_k_per_part, device=partid.device)
            dino_feats = self.dino_embeddings(kids.long())
            output = self.dino_mapper(dino_feats).reshape(kids.size(0), -1)
            logvar = None

            mus = []
            for i in range(partid.size(0)):
                mu = output[
                    :, partid[i] * self.vae_dims : (partid[i] + 1) * self.vae_dims
                ]
                mus.append(mu)
            # Mx(CxD)
            output = torch.stack(mus, dim=0)  # Mx(CxD)
        else:
            kids = (
                torch.arange(0, self.num_k_per_part, device=partid.device)
                .unsqueeze(0)
                .repeat(len(partid), 1)
            )
            output = self.embedding(
                partid.unsqueeze(1) * (self.num_k_per_part + 1) + kids
            )  # (M, C, D)

        if covariance:
            sampled = []

            for m in range(len(output)):
                weights = output[m]  # (N, D)
                mean = weights.mean(0)
                weights = weights - mean[None, :]
                cov = torch.einsum("nd,nc->dc", weights, weights) / (
                    weights.shape[0] - 1
                )
                dist = distributions.multivariate_normal.MultivariateNormal(
                    mean, covariance_matrix=cov
                )
                z_init = dist.sample((batch_size,))
                sampled.append(z_init)

            sampled = torch.stack(sampled, dim=1)  # (B, M, D)
        else:
            var, mean = torch.var_mean(output, dim=1)  # (M, D)
            var = var.unsqueeze(0).repeat(batch_size, 1, 1)
            mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
            sampled = torch.randn_like(mean) * torch.sqrt(var) + mean

        return sampled

    def forward(
        self,
        hashes,
        index: Optional[torch.Tensor] = None,
        get_mu=False,
        input_is_mu=False,
    ):
        B = hashes.size(0)

        if input_is_mu:
            orig_mu = hashes  # (B, N, d)
        else:
            # 0, 257, 514, ...
            if index is None:
                offset = torch.arange(self.num_parts, device=hashes.device) * (
                    self.num_k_per_part + 1
                )
                orig_mu = self.embedding(
                    hashes.long() + offset.reshape(1, -1)
                )  # (B, N, d)
            else:
                offset = index.reshape(-1) * (self.num_k_per_part + 1)
                orig_mu = self.embedding(
                    hashes.long() + offset.reshape(B, -1).long()
                )  # (B, N, d)

        if index is not None:
            pe = self.pe[index.reshape(-1)]  # index must be equal size
            pe = pe.reshape(B, -1, pe.shape[-1])
            mu = orig_mu  # + pe
        else:
            pe = self.pe.unsqueeze(0).repeat(B, 1, 1)
            mu = orig_mu  # + pe

        if self.concat_pe:
            mu = torch.cat([mu, pe], dim=-1)
            projected = self.projection(mu)
        else:
            projected = self.projection(mu) + pe

        if get_mu:
            return projected, orig_mu, pe

        return projected

    def get_dummy_dino_features(self, batch_size=1):
        # for partimagenet only!
        return torch.randn(1, 768).repeat(batch_size, 1)


class PartCraftTokenMapper(nn.Module):
    def __init__(
        self,
        num_parts,
        num_k_per_part,
        out_dims,
        projection_nlayers=1,
        projection_activation=nn.ReLU(),
        with_pe=True,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.num_k_per_part = num_k_per_part
        self.with_pe = with_pe
        self.out_dims = out_dims

        self.embedding = nn.Embedding((self.num_k_per_part + 1) * num_parts, out_dims)
        if with_pe:
            self.pe = nn.Parameter(torch.randn(num_parts, out_dims))
        else:
            self.register_buffer("pe", torch.zeros(num_parts, out_dims))

        if projection_nlayers == 0:
            self.projection = nn.Identity()
        else:
            projections = []
            for i in range(projection_nlayers - 1):
                projections.append(nn.Linear(out_dims, out_dims))
                projections.append(projection_activation)

            projections.append(nn.Linear(out_dims, out_dims))
            self.projection = nn.Sequential(*projections)

    def obtain_code_mu(self, partid, kid, *args, **kwargs):
        # partid: torch.tensor([0,1,2])
        # kid: torch.tensor([16,16,16])
        # output: torch.tensor(...) N x D

        output = self.embedding(partid * (self.num_k_per_part + 1) + kid)
        return output

    def get_dummy_dino_features(self, batch_size=1):
        # for partimagenet only!
        return torch.randn(1, 768).repeat(batch_size, 1)

    def random_sample_mu(self, batch_size, partid=None, **kwargs):
        # partid: torch.tensor([0,1,2])
        if partid is None:
            partid = torch.arange(
                0, self.num_parts, device=self.embedding.weight.device
            )

        kids = (
            torch.arange(0, self.num_k_per_part, device=partid.device)
            .unsqueeze(0)
            .repeat(len(partid), 1)
        )
        output = self.embedding(
            partid.unsqueeze(1) * (self.num_k_per_part + 1) + kids
        )  # (M, C, D)
        var, mean = torch.var_mean(output, dim=1)  # (M, D)
        var = var.unsqueeze(0).repeat(batch_size, 1, 1)
        mean = mean.unsqueeze(0).repeat(batch_size, 1, 1)
        sampled = torch.randn_like(mean) * torch.sqrt(var) + mean
        return sampled

    def forward(self, hashes, index: Optional[torch.Tensor] = None, input_is_mu=False):
        B = hashes.size(0)

        if not input_is_mu:
            # 0, 257, 514, ...
            if index is None:
                offset = torch.arange(self.num_parts, device=hashes.device) * (
                    self.num_k_per_part + 1
                )
                hashes = self.embedding(
                    hashes.long() + offset.reshape(1, -1)
                )  # (B, N, d)
            else:
                offset = index.reshape(-1) * (self.num_k_per_part + 1)
                hashes = self.embedding(
                    hashes.long() + offset.reshape(B, -1).long()
                )  # (B, N, d)

        if index is not None:
            pe = self.pe[index.reshape(-1)]  # index must be equal size
            pe = pe.reshape(B, -1, self.out_dims)
            hashes = hashes + pe
        else:
            hashes = hashes + self.pe.unsqueeze(0).repeat(B, 1, 1)
        projected = self.projection(hashes)

        return projected
