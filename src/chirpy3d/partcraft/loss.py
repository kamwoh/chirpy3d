import copy
import random
from math import prod

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as Ft
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention

from partcraft.dino import DINO
from partcraft.kmeans_segmentation import KMeansSegmentation


def _compute_self_attn_loss(batch, self_attn_probs, attn_size, loss_method="mean"):
    H = W = attn_size
    avg_attn_map = []
    for name in self_attn_probs:
        avg_attn_map.append(self_attn_probs[name])
    avg_attn_map = torch.stack(avg_attn_map, dim=0).mean(
        dim=0
    )  # (L,B,H*W,H*W) -> (B,H*W,H*W)

    segmasks = (
        F.interpolate(batch["masks"].float(), (H, W), mode="nearest") > 0.5
    )  # (B, M, H, W)

    loss = 0
    count = 0

    max_selfattn = torch.tensor(0.0)

    for bi in range(segmasks.size(0)):
        for mi in range(segmasks.size(1)):
            segmask = segmasks[bi, mi].reshape(-1)
            if segmask.sum().item() == 0:
                continue

            if segmask.sum().item() == prod(
                segmask.size()
            ):  # whole image is the mask, skip
                continue

            self_attn_outside = avg_attn_map[bi, segmask][:, ~segmask]
            if loss_method == "mean":  # this is from attention refocus paper
                loss += self_attn_outside.mean()
            elif loss_method == "minmax":
                self_attn_outside_mean = self_attn_outside.sum(
                    dim=1
                )  # (S, S') -> (S, )

                self_attn_inside = avg_attn_map[bi, segmask][:, segmask]
                self_attn_inside_mean = self_attn_inside.sum(dim=1)  # (S, S) -> (S, )
                loss += (
                    self_attn_outside_mean / self_attn_inside_mean
                ).mean()  # maximize inside, minimize outside
            else:
                self_attn_outside_outside = avg_attn_map[bi, ~segmask][:, ~segmask].sum(
                    dim=1
                )
                self_attn_outside_inside = avg_attn_map[bi, ~segmask][:, segmask].sum(
                    dim=1
                )
                self_attn_inside_outside = avg_attn_map[bi, segmask][:, ~segmask].sum(
                    dim=1
                )
                self_attn_inside_inside = avg_attn_map[bi, segmask][:, segmask].sum(
                    dim=1
                )

                outside_loss = (
                    self_attn_outside_inside / self_attn_outside_outside
                ).mean() * 0.5
                inside_loss = (
                    self_attn_inside_outside / self_attn_inside_inside
                ).mean() * 0.5

                # min outside outside, max outside inside, min inside outside, max inside inside
                loss += outside_loss + inside_loss

            count += 1

            max_selfattn = (
                self_attn_outside.max()
                if self_attn_outside.max() > max_selfattn
                else max_selfattn
            )

    return loss / count, max_selfattn


def _compute_avg_attn_map(attn_probs):
    avg_attn_map = []
    for name in attn_probs:
        avg_attn_map.append(attn_probs[name])

    # average over layers
    avg_attn_map = torch.stack(avg_attn_map, dim=0).mean(
        dim=0
    )  # (L,B,H,W,77) -> (B,H,W,77)
    return avg_attn_map


def _compute_attn_loss_sub(
    batch,
    located_attn_map,
    shape_attn_loss=False,
    background_attn_loss_scale=1.0,
    normalize=True,
    attn_scale=1.0,
    exp_attn=False,
    gs_blur_attn=False,
):
    H, W = located_attn_map.size()[-2:]
    segmasks = batch["masks"].float()

    if gs_blur_attn:
        segmasks = Ft.gaussian_blur(segmasks, [3, 3], [1, 1])
        segmasks = F.interpolate(segmasks, (H, W), mode="bilinear").to(
            located_attn_map.dtype
        )
    else:
        segmasks = F.interpolate(segmasks, (H, W), mode="nearest").to(
            located_attn_map.dtype
        )

    # it is possible that a location is attended by two or more tokens, this depends on the resize above
    numseg_per_region = segmasks.sum(dim=1, keepdims=True)
    # if normalize:
    #     segmasks = segmasks / numseg_per_region.clamp(min=1)  # prevent zero division

    # if that region has no label at all, assign equal weight to all tokens
    # happens only when shape attn loss is used, else all regions will have at least one token
    # if shape attn loss is used, background token will be removed, hence no label
    # then token should not attend these regions --- having 1 / M as target
    allzero_regions = (numseg_per_region == 0).float().squeeze(1)
    # segmasks = allzero_regions * 1 / M + (1 - allzero_regions) * segmasks

    # normalized along all sub-concepts
    if normalize:
        _located_attn_map = located_attn_map * attn_scale
        if exp_attn:
            _located_attn_map = torch.exp(_located_attn_map)
        norm_map = _located_attn_map / _located_attn_map.sum(dim=1, keepdim=True).clamp(
            min=1e-6
        )  # prevent zero division
    else:
        norm_map = located_attn_map

    # version 1 loss
    # if norm_map is assigned manually, means the sub-concept token is not found, hence no gradient will be backprop
    # attn_loss = F.binary_cross_entropy(norm_map.clamp(min=0, max=1),
    #                                    segmasks.clamp(min=0, max=1), reduction='none').mean(dim=(1, 2, 3))
    # attn_loss = F.binary_cross_entropy(norm_map.clamp(min=0, max=1),
    #                                    segmasks.clamp(min=0, max=1), reduction='none')

    # version 2 loss, softmax loss, the attention map is "predicting" the index of the mask
    attn_loss_object = -(segmasks * torch.log(norm_map.clamp(min=1e-6))).sum(
        dim=1
    )  # (B, H, W)
    if shape_attn_loss:
        # token should not attend these regions, minimizing their summed attention
        region_sum_prob = 1 - located_attn_map.sum(dim=1).clamp(min=1e-6)  # (B, H, W)
        attn_loss_background = -(
            allzero_regions * torch.log(region_sum_prob)
        )  # (B, H, W)
        attn_loss = (
            attn_loss_object + attn_loss_background * background_attn_loss_scale
        ).mean(dim=(1, 2))
    else:
        attn_loss = attn_loss_object.mean(dim=(1, 2))

    is_extra = batch.get("extra", torch.zeros_like(attn_loss))

    attn_loss = attn_loss * (1 - is_extra)  # filter out extra data
    attn_loss = attn_loss.sum() / (1 - is_extra).sum().clamp(
        min=1
    )  # clamp at 1 in case zero division

    return attn_loss


def _compute_attn_loss(
    batch,
    attn_probs,
    placeholder_token_ids,
    accelerator,
    max_attn=1,
    shape_attn_loss=False,
    background_attn_loss_scale=1.0,
    normalize=True,
    attn_scale=1.0,
    exp_attn=False,
    gs_blur_attn=False,
):
    avg_attn_map = _compute_avg_attn_map(attn_probs)
    B, H, W, seq_length = avg_attn_map.size()
    located_attn_map = []

    # locate the attn map
    for i, placeholder_token_id in enumerate(placeholder_token_ids):
        for bi in range(B):
            if "input_ids" in batch:
                learnable_idx = (
                    batch["input_ids"][bi] == placeholder_token_id
                ).nonzero(as_tuple=True)[0]
            else:
                learnable_idx = (
                    batch["input_ids_one"][bi] == placeholder_token_id
                ).nonzero(as_tuple=True)[0]

            if len(learnable_idx) != 0:  # only assign if found
                if len(learnable_idx) == 1:
                    offset_learnable_idx = learnable_idx
                else:  # if there is two and above.
                    raise NotImplementedError

                located_map = avg_attn_map[bi, :, :, offset_learnable_idx]
                located_attn_map.append(located_map)
            else:
                located_attn_map.append(torch.zeros(H, W, 1).to(accelerator.device))

    M = len(placeholder_token_ids)
    located_attn_map = (
        torch.stack(located_attn_map, dim=0).reshape(M, B, H, W).transpose(0, 1)
    )  # (B, M, 16, 16)
    located_attn_map = located_attn_map.clamp(max=max_attn)

    attn_loss = _compute_attn_loss_sub(
        batch,
        located_attn_map,
        shape_attn_loss,
        background_attn_loss_scale,
        normalize,
        attn_scale,
        exp_attn,
        gs_blur_attn,
    )

    return attn_loss, located_attn_map.detach().max(), located_attn_map


def multi_resolutions_attn_loss_fn(
    batch, unet, placeholder_token_ids, accelerator, attn_sizes, max_attn=1
):
    attn_probs = {}

    for attn_size in attn_sizes:
        attn_probs[attn_size] = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and module.attn_probs is not None:
            a = module.attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
            attn_size = a.size(1)
            attn_probs[attn_size][name] = a

    total_loss = 0
    avg_max = 0
    for attn_size in attn_sizes:
        attn_loss, max_attn_val, _ = _compute_attn_loss(
            batch, attn_probs[attn_size], placeholder_token_ids, accelerator, max_attn
        )
        total_loss += attn_loss
        avg_max += max_attn_val

    total_loss /= len(attn_sizes)
    avg_max /= len(attn_sizes)
    return total_loss, avg_max


def attn_loss_fn(batch, unet, placeholder_token_ids, accelerator, max_attn=1):
    attn_probs = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention) and module.attn_probs is not None:
            a = module.attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
            attn_probs[name] = a

    return _compute_attn_loss(
        batch, attn_probs, placeholder_token_ids, accelerator, max_attn
    )


def dreamcreature_loss(
    batch,
    unet: UNet2DConditionModel,
    dino: DINO,
    seg: KMeansSegmentation,
    placeholder_token_ids,
    accelerator,
    attn_size=16,
    selfattn_loss=False,
    selfattn_loss_method="mean",
    shape_attn_loss=False,
    background_attn_loss_scale=1.0,
    sampled_t=None,  # (B,)
    shape_attn_t_threshold=1000,
    drop_tokens=False,
    drop_rate=0.5,
    normalize=True,
    attn_scale=1.0,
):
    attn_sizes = attn_size
    if isinstance(attn_sizes, int):
        attn_sizes = [attn_size]

    attn_probs = {}
    self_attn_probs = {}

    for attn_size in attn_sizes:
        attn_probs[attn_size] = {}
        self_attn_probs[attn_size] = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            if module.attn_probs is not None:
                a = module.attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
                attn_size = a.size(1)
                attn_probs[attn_size][name] = a
            elif module.self_attn_probs is not None:
                a = module.self_attn_probs.mean(
                    dim=1
                )  # (B,Head,H*W,H*W) -> (B,H*W,H*W)
                attn_size = int(a.size(1) ** 0.5)
                self_attn_probs[attn_size][name] = a

    raw_images = batch["raw_images"]
    dino_input = dino.preprocess(raw_images, size=224).to(accelerator.device)
    with torch.no_grad():
        dino_ft = dino.get_feat_maps(dino_input)
        segmasks, appeared_tokens, fg_mask = seg.get_segmask(
            dino_ft.float(),
            with_appeared_tokens=True,
            exclude_bg=shape_attn_loss,
            get_fg_mask=True,
        )  # (B, M, H, W)
        masks = []
        for i, appeared in enumerate(appeared_tokens):  # loop B times
            if drop_tokens and random.random() < drop_rate:
                appeared = copy.copy(appeared)
                random.shuffle(appeared)
                appeared = appeared[: len(appeared) // 2]  # randomly drop half
            mask = (segmasks[i, appeared].sum(dim=0) > 0).float()  # (A, H, W) -> (H, W)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)  # (B, H, W)

        if shape_attn_loss:
            fg_codes = list(range(segmasks.size(1)))
            # remove background code from fg_codes
            fg_codes.remove(seg.background_code)

            for i, t in enumerate(sampled_t):
                if t >= shape_attn_t_threshold:
                    segmasks[i, seg.background_code] = fg_mask[
                        i
                    ]  # replace background with the foreground mask
                    segmasks[i, fg_codes] = 0  # remove other foreground masks
                else:  # always 0, not allowing the shape token to attend to these region
                    segmasks[i, seg.background_code] = 0

        batch["masks"] = segmasks  # for attention loss

    total_loss = 0
    avg_max = 0
    avg_self_max = 0
    for attn_size in attn_sizes:
        attn_loss, max_attn_val, _ = _compute_attn_loss(
            batch,
            attn_probs[attn_size],
            placeholder_token_ids,
            accelerator,
            shape_attn_loss=shape_attn_loss,
            background_attn_loss_scale=background_attn_loss_scale,
            normalize=normalize,
            attn_scale=attn_scale,
        )
        total_loss += attn_loss
        avg_max += max_attn_val

        if selfattn_loss:
            self_attn_loss, max_self_attn_val = _compute_self_attn_loss(
                batch, self_attn_probs[attn_size], attn_size, selfattn_loss_method
            )
            total_loss += self_attn_loss
            avg_self_max += max_self_attn_val

    batch["image_masks"] = masks  # update for later

    total_loss /= len(attn_sizes)
    avg_max /= len(attn_sizes)
    if selfattn_loss:
        avg_self_max /= len(attn_sizes)
        return total_loss, (avg_max, avg_self_max)

    return total_loss, avg_max


def dreamcreature_loss_v2(
    batch,
    unet: UNet2DConditionModel,
    placeholder_token_ids,
    accelerator,
    attn_size=16,
    selfattn_loss=False,
    selfattn_loss_method="mean",
    shape_attn_loss=False,
    background_attn_loss_scale=1.0,
    normalize=True,
    attn_scale=1.0,
    exp_attn=False,
    gs_blur_attn=False,
    store_embeddings=False,
):
    attn_sizes = attn_size
    if isinstance(attn_sizes, int):
        attn_sizes = [attn_size]

    attn_probs = {}
    self_attn_probs = {}
    embeddings = {}

    for attn_size in attn_sizes:
        attn_probs[attn_size] = {}
        self_attn_probs[attn_size] = {}
        embeddings[attn_size] = {}

    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            if module.attn_probs is not None:
                a = module.attn_probs.mean(dim=1)  # (B,Head,H,W,77) -> (B,H,W,77)
                attn_size = a.size(1)
                attn_probs[attn_size][name] = a
            elif module.self_attn_probs is not None:
                a = module.self_attn_probs.mean(
                    dim=1
                )  # (B,Head,H*W,H*W) -> (B,H*W,H*W)
                attn_size = int(a.size(1) ** 0.5)
                self_attn_probs[attn_size][name] = a

            if store_embeddings and module.embeddings is not None:
                e = module.embeddings  # (B, HW, C)
                attn_size = int(e.size(1) ** 0.5)
                if attn_size not in embeddings:
                    embeddings[attn_size] = {}
                embeddings[attn_size][name] = e

    if store_embeddings:
        batch["embeddings"] = embeddings

    total_loss = 0
    avg_max = 0
    avg_self_max = 0
    batch["located_attn_map"] = {}

    for attn_size in attn_sizes:
        attn_loss, max_attn_val, located_attn_map = _compute_attn_loss(
            batch,
            attn_probs[attn_size],
            placeholder_token_ids,
            accelerator,
            shape_attn_loss=shape_attn_loss,
            background_attn_loss_scale=background_attn_loss_scale,
            normalize=normalize,
            attn_scale=attn_scale,
            exp_attn=exp_attn,
            gs_blur_attn=gs_blur_attn,
        )
        total_loss += attn_loss
        avg_max += max_attn_val

        batch["located_attn_map"][attn_size] = located_attn_map

        if selfattn_loss:
            self_attn_loss, max_self_attn_val = _compute_self_attn_loss(
                batch, self_attn_probs[attn_size], attn_size, selfattn_loss_method
            )
            total_loss += self_attn_loss
            avg_self_max += max_self_attn_val

    total_loss /= len(attn_sizes)
    avg_max /= len(attn_sizes)
    if selfattn_loss:
        avg_self_max /= len(attn_sizes)
        return total_loss, (avg_max, avg_self_max)

    return total_loss, avg_max
