# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import copy
import logging
import math
import os
import random
import shutil
from pathlib import Path

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import yaml
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_snr
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from mvdream_unet import get_mvdream_unet, MVDreamUNet2DConditionModel
from mvdream_utils import get_camera
from partcraft.dataset import DreamCreatureDataset
from partcraft.dino import DINO
from partcraft.kmeans_segmentation import KMeansSegmentation
from partcraft.loss import dreamcreature_loss_v2
from partcraft.pipeline import DreamCreatureSDPipeline
from partcraft.text_encoder import CustomCLIPTextModel
from partcraft.tokenizer import MultiTokenCLIPTokenizer
from utils import (
    add_tokens,
    tokenize_prompt,
    get_attn_processors,
    setup_attn_processor,
    load_attn_processor,
    setup_token_mapper,
)

imagenet_templates = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def save_model_card(
    repo_id: str, images=None, base_model=str, dataset_name=str, repo_folder=None
):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA text2image fine-tuning - {repo_id}
These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1-base",  # use 2.1 base by default
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is sampled during training for inference.",
    )
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub.",
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )

    ### all belows are my custom arguments, some of them are experimental during this research ###
    parser.add_argument("--filename", default="train.txt")
    parser.add_argument("--code_filename", default="train_caps_better_m8_k256.txt")
    parser.add_argument("--repeat", default=1, type=int)

    parser.add_argument(
        "--scheduler_steps",
        default=1000,
        type=int,
        help="scheduler step, if turbo, set to 4",
    )
    parser.add_argument("--num_parts", type=int, default=4, help="Number of parts")
    parser.add_argument("--num_k_per_part", type=int, default=256, help="Number of k")

    parser.add_argument("--mapper_lr_scale", default=1, type=float)
    parser.add_argument("--mapper_lr", default=0.0001, type=float)
    parser.add_argument("--attn_loss", default=0, type=float)
    parser.add_argument("--projection_nlayers", default=3, type=int)
    parser.add_argument(
        "--projection_actfn", default="relu", choices=["relu", "silu", "sine"]
    )

    parser.add_argument("--masked_training", action="store_true")
    parser.add_argument("--drop_tokens", action="store_true")
    parser.add_argument("--drop_rate", type=float, default=0.5)
    parser.add_argument("--drop_counts", default="half")

    parser.add_argument("--class_name", default="")
    parser.add_argument("--no_pe", action="store_true")
    parser.add_argument("--vector_shuffle", action="store_true")
    parser.add_argument("--use_templates", action="store_true")

    parser.add_argument("--use_gt_label", action="store_true")
    parser.add_argument("--bg_code", default=7, type=int)  # for gt_label
    parser.add_argument("--fg_idx", default=0, type=int)  # for gt_label

    parser.add_argument("--filter_class", default=None, help="debugging purpose")

    parser.add_argument("--attn_size", default=None)

    parser.add_argument("--vae_dims", default=0, type=int)
    parser.add_argument("--kl_loss", default=0, type=float)

    parser.add_argument("--gs", default=7.5, type=float)

    parser.add_argument(
        "--extra_data_dir", default=None, help="this is for data without mask"
    )

    parser.add_argument("--selfattn_loss", default=False, action="store_true")
    parser.add_argument(
        "--selfattn_loss_method",
        default="mean",
        choices=["mean", "minmax", "minmax_all"],
    )
    parser.add_argument(
        "--shape_attn_loss",
        default=False,
        action="store_true",
        help="if it is enable, we ignore background, and reuse background code as shape code,"
        "not that this must be using ground truth label",
    )
    parser.add_argument(
        "--background_attn_loss_scale",
        default=0.0,
        type=float,
        help="background attn loss scale",
    )
    parser.add_argument(
        "--t_threshold", default=900, type=int, help="threshold for shape attn loss"
    )
    parser.add_argument("--dino_features", default=False, action="store_true")

    parser.add_argument("--recaption", default=False, action="store_true")
    parser.add_argument("--concat_pe", default=False, action="store_true")
    parser.add_argument(
        "--learnable_dino_embeddings", default=False, action="store_true"
    )
    parser.add_argument("--skip_dino_features", default=False, action="store_true")
    parser.add_argument("--ema_rate", default=0.9, type=float)
    parser.add_argument("--dino_nlayers", default=2, type=int)

    parser.add_argument("--no_attn_norm", default=False, action="store_true")
    parser.add_argument("--skip_background", default=False, action="store_true")

    parser.add_argument(
        "--attn_scale", default=1.0, type=float, help="only for normalized attn"
    )
    parser.add_argument(
        "--exp_attn",
        default=False,
        action="store_true",
        help="only for normalized attn",
    )
    parser.add_argument(
        "--gs_blur_attn", default=False, action="store_true", help="gs blur for attn"
    )

    parser.add_argument(
        "--use_new_lora",
        default=False,
        action="store_true",
        help="wanted to adapt to latest diffusers but i am lazy, sorry",
    )
    parser.add_argument(
        "--shared_shape_token",
        default=False,
        action="store_true",
        help="shared shape token; can ignore as it is experimental codes",
    )

    parser.add_argument("--mixed_latent_reg", default=0.0, type=float)

    parser.add_argument("--skip_sa", default=False, action="store_true")

    parser.add_argument(
        "--mode", default="weighted", choices=["spatial", "weighted", "mean", "topk"]
    )
    parser.add_argument("--topk", default=5, type=int)
    parser.add_argument("--add_reg_attn", default=False, action="store_true")
    parser.add_argument("--init_lora_weights", default=False, action="store_true")

    parser.add_argument("--part_craft_token_mapper", default=False, action="store_true")

    parser.add_argument("--partids", default="0,1,2,4,6", type=str)
    parser.add_argument("--remove_unused", default=False, action="store_true")

    parser.add_argument(
        "--enable_cfg", default=False, action="store_true", help="still experimental"
    )

    parser.add_argument("--prompt_dropout", default=0.0, type=float)

    parser.add_argument("--enable_grad_for_mu", default=False, action="store_true")

    parser.add_argument("--no_learn_mu", default=False, action="store_true")

    parser.add_argument("--learn_whole", default=False, action="store_true")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    return args


def collate_fn(args, tokenizer, placeholder_token):
    resize = args.resolution

    if "sims4" in args.train_data_dir:
        resize = args.resolution * (256 / 224)

    train_resizecrop = transforms.Compose(
        [
            transforms.Resize(int(resize), InterpolationMode.BILINEAR),
            transforms.RandomCrop(args.resolution),
        ]
    )

    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def f(examples):
        raw_images = [train_resizecrop(example["pixel_values"]) for example in examples]

        pixel_values = torch.stack([train_transforms(image) for image in raw_images])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        captions = []
        appeared_tokens = []

        for i in range(len(examples)):
            if args.use_templates and random.random() <= 0.5:  # 50% using templates
                if args.class_name != "":
                    caption = random.choice(imagenet_templates).format(
                        f"{placeholder_token} {args.class_name}"
                    )
                else:
                    caption = random.choice(imagenet_templates).format(
                        placeholder_token
                    )
            else:
                if args.class_name != "":
                    caption = f"{placeholder_token} {args.class_name}"
                else:
                    caption = placeholder_token

            tokens = tokenizer.token_map[placeholder_token][: args.num_parts]
            tokens = [tokens[a] for a in examples[i]["appeared"]]

            if args.vector_shuffle or args.drop_tokens:
                tokens = copy.copy(tokens)
                random.shuffle(tokens)

            if not args.enable_cfg:
                if (
                    args.drop_tokens
                    and random.random() < args.drop_rate
                    and len(tokens) >= 2
                ):
                    # randomly drop half of the tokens
                    if args.drop_counts == "half":
                        tokens = tokens[: len(tokens) // 2]
                    else:
                        tokens = tokens[: int(args.drop_counts)]

            appeared = [int(t.split("_")[1]) for t in tokens]  # <part>_i
            appeared_tokens.append(appeared)

            if random.random() < args.prompt_dropout:
                caption = ""
            else:
                caption = caption.replace(placeholder_token, " ".join(tokens))
            captions.append(caption)

        input_ids = tokenize_prompt(tokenizer, captions)
        # input_ids = inputs.input_ids.repeat(len(examples), 1)  # (1, 77) -> (B, 77)

        codes = torch.stack([example["codes"] for example in examples])
        extras = torch.stack([example["extra"] for example in examples])
        labels = torch.stack([example["labels"] for example in examples])

        return {
            "pixel_values": pixel_values,
            "raw_images": raw_images,
            "appeared_tokens": appeared_tokens,
            "input_ids": input_ids,
            "codes": codes,
            "extra": extras,
            "labels": labels,
            "captions": captions,
        }

    return f


def recaption(
    appeared_tokens,
    tokenizer,
    vector_shuffle=True,
    use_templates=True,
    placeholder_token="<part>",
    drop_tokens=False,
    drop_rate=0.5,
    background_code=7,
    skip_background=False,
    enable_cfg=False,
    orig_captions=None,
):
    captions = []
    appeared_tokens_ret = []

    for i in range(len(appeared_tokens)):
        if use_templates and random.random() <= 0.5:  # 50% using templates
            caption = random.choice(imagenet_templates).format(placeholder_token)
        else:
            caption = placeholder_token

        tokens = tokenizer.token_map[placeholder_token]
        tokens = [
            tokens[a] for a in appeared_tokens[i]
        ]  # appeared_tokens from segmentation map

        if vector_shuffle:
            tokens = copy.copy(tokens)
            random.shuffle(tokens)

        if not enable_cfg:
            if drop_tokens and random.random() < drop_rate and len(tokens) >= 2:
                # randomly drop half of the tokens
                tokens = tokens[: len(tokens) // 2]

        if (
            placeholder_token + f"_{background_code}" not in tokens
            and not skip_background
        ):
            tokens.append(placeholder_token + f"_{background_code}")

        filtered_appeared_tokens = [int(t.split("_")[1]) for t in tokens]  # <part>_i
        appeared_tokens_ret.append(filtered_appeared_tokens)

        if orig_captions is not None and orig_captions[i] == "":
            caption = ""
        else:
            caption = caption.replace(placeholder_token, " ".join(tokens))
        captions.append(caption)

    return captions, appeared_tokens_ret


def obtain_segmentation_mask_(
    batch,
    dino: DINO,
    seg: KMeansSegmentation,
    accelerator,
    shape_attn_loss=False,
    sampled_t=None,  # (B,)
    shape_attn_t_threshold=1000,
    perform_recaption=False,
    drop_tokens=False,
    drop_rate=0.5,
    tokenizer=None,
    vector_shuffle=False,
    use_templates=True,
    placeholder_token="<part>",
    skip_background=False,
    shared_shape_token=False,
    num_k_per_part=256,
    map_idxs=tuple(),
    bg_code=7,
    enable_cfg=False,
    learn_whole=False,
):
    raw_images = batch["raw_images"]
    dino_input = dino.preprocess(raw_images, size=224).to(accelerator.device)
    with torch.no_grad():
        dino_ft, dino_cls = dino.get_feat_maps(dino_input, get_cls_token=True)
        segmasks, appeared_tokens, fg_mask = seg.get_segmask(
            dino_ft.float(),
            with_appeared_tokens=True,
            exclude_bg=shape_attn_loss,
            get_fg_mask=True,
            map_idxs=map_idxs,
        )  # (B, M, H, W)

        if learn_whole:
            appeared_tokens = [a + [seg.M] for a in appeared_tokens]

        if skip_background:
            for a in appeared_tokens:
                if bg_code in a:
                    a.remove(bg_code)

            segmasks[:, bg_code] = 0

        if perform_recaption:
            captions, appeared_tokens = recaption(
                appeared_tokens,
                tokenizer,
                vector_shuffle or drop_tokens,
                use_templates,
                placeholder_token,
                drop_tokens,
                drop_rate,
                bg_code,
                skip_background,
                enable_cfg,
                batch["captions"],
            )
            input_ids = tokenize_prompt(tokenizer, captions)

            codes = batch["codes"]
            if learn_whole:
                codes = torch.cat([codes, torch.ones_like(codes)[:, :1]], dim=1)
            codes[:, :] = num_k_per_part
            for c, l, a in zip(codes, batch["labels"].to(codes), appeared_tokens):
                c[a] = l
                if shared_shape_token:  # invariant shape
                    c[bg_code] = num_k_per_part

            batch["codes"] = codes
            if enable_cfg:
                rand_cfg_mask = (torch.rand_like(codes) < 0.2).float()
                batch["codes"] = (
                    codes * (1 - rand_cfg_mask) + rand_cfg_mask * num_k_per_part
                )

            batch["input_ids"] = input_ids.to(batch["input_ids"])

        masks = []
        for i, appeared in enumerate(appeared_tokens):  # loop B times
            appeared = sorted(appeared)
            if learn_whole and appeared[-1] == seg.M:
                mask = (
                    segmasks[i, appeared[:-1]].sum(dim=0) > 0
                ).float()  # (A, H, W) -> (H, W)
            else:
                mask = (
                    segmasks[i, appeared].sum(dim=0) > 0
                ).float()  # (A, H, W) -> (H, W)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)  # (B, H, W)

        if shape_attn_loss:
            fg_codes = list(range(segmasks.size(1)))
            # remove background code from fg_codes
            fg_codes.remove(bg_code)

            for i, t in enumerate(sampled_t):
                if t >= shape_attn_t_threshold:
                    segmasks[i, bg_code] = fg_mask[
                        i
                    ]  # replace background with the foreground mask
                    segmasks[i, fg_codes] = 0  # remove other foreground masks
                else:  # always 0, not allowing the shape token to attend to these region
                    segmasks[i, bg_code] = 0

        batch["masks"] = segmasks  # for attention loss
        batch["image_masks"] = masks  # for masked diffusion loss
        batch["dino_cls"] = dino_cls


def obtain_embeddings(batch, mode="weighted", k=5):
    attn_sizes = batch["embeddings"].keys()
    embeddings = {}

    for attn_size in attn_sizes:
        e = batch["embeddings"][attn_size]  # (B, HW, C) -> (B, C, HW)
        e = torch.cat([e[name].permute(0, 2, 1) for name in e], dim=1)  # concat at C
        e_flatten = e.flatten(2)  # (B, C, H*W)

        if mode != "spatial":
            a = batch["located_attn_map"][attn_size]  # (B, M, H, W)
            a_flatten = a.flatten(2).to(e_flatten)  # (B, M, H*W)

        if mode == "spatial":
            e = e_flatten  # (B, C, H*W)
        elif mode == "weighted":
            e = torch.einsum("bch,bmh->bcm", e_flatten, a_flatten)  # (B, C, M)
        elif mode == "topk":
            a_topk = torch.topk(a_flatten, k, dim=-1).values  # (B, M, K)
            a_topk = a_topk / (a_topk.sum(dim=-1, keepdim=True) + 1e-6)
            e = torch.einsum("bch,bmk->bcm", e_flatten, a_topk)  # (B, C, M)
        elif mode == "mean":
            e = e_flatten.mean(dim=-1)
        else:
            raise NotImplementedError

        embeddings[attn_size] = e

    return embeddings


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training."
            )
        import wandb

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            ).repo_id
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )

    OUT_DIMS = (
        1024
        if "stabilityai/stable-diffusion-2-1" in args.pretrained_model_name_or_path
        else 768
    )

    text_encoder = CustomCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
    )

    unet: MVDreamUNet2DConditionModel = get_mvdream_unet()

    dino = DINO()
    seg = KMeansSegmentation(
        args.train_data_dir + "/pretrained_kmeans.pth",
        args.fg_idx,
        args.bg_code,
        8,
        256,
    )

    simple_mapper = setup_token_mapper(args, OUT_DIMS)

    # initialize placeholder token
    placeholder_token = "<part>"
    initializer_token = None
    additional = 1 if args.learn_whole else 0
    placeholder_token_ids = add_tokens(
        tokenizer,
        text_encoder,
        placeholder_token,
        args.num_parts + additional,
        initializer_token,
    )

    # freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    dino.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    dino.to(accelerator.device)

    # now we will add new LoRA weights to the attention layers
    # It's important to realize here how many attention weights will be added and of which sizes
    # The sizes of the attention layers consist only of two different variables:
    # 1) - the "hidden_size", which is increased according to `unet.config.block_out_channels`.
    # 2) - the "cross attention size", which is set to `unet.config.cross_attention_dim`.

    # Let's first see how many attention processors we will have to set.
    # For Stable Diffusion, it should be equal to:
    # - down blocks (2x attention layers) * (2x transformer layers) * (3x down blocks) = 12
    # - mid blocks (2x attention layers) * (1x transformer layers) * (1x mid blocks) = 2
    # - up blocks (2x attention layers) * (3x transformer layers) * (3x down blocks) = 18
    # => 32 layers

    # Set correct lora layers
    attn_size = args.attn_size
    if attn_size is None or attn_size == "":
        attn_size = [args.resolution // 32]  # default
    else:
        attn_size = [int(a) for a in attn_size.split(",")]

    tag_embeddings = args.mixed_latent_reg != 0

    setup_attn_processor(
        unet,
        rank=args.rank,
        attn_size=attn_size,
        resolution=args.resolution,
        selfattn_loss=args.selfattn_loss,
        tag_embeddings=tag_embeddings,
        skip_sa=args.skip_sa,
        init_lora_weights=args.init_lora_weights,
    )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    lora_layers = AttnProcsLayers(get_attn_processors(unet))

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    if args.no_learn_mu:
        simple_mapper.embedding.requires_grad_(False)

    extra_params = list(simple_mapper.parameters())
    mapper_lr = (
        args.learning_rate * args.mapper_lr_scale
        if args.learning_rate != 0
        else args.mapper_lr
    )

    optimizer = optimizer_cls(
        [
            {"params": lora_layers.parameters()},
            {"params": extra_params, "lr": mapper_lr},
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    keep_partids = tuple()
    if args.remove_unused:
        # because some clusters are not used
        keep_partids = tuple(
            [seg.background_code] + list(map(int, args.partids.split(",")))
        )

    train_dataset = DreamCreatureDataset(
        args.train_data_dir,
        args.filename,
        code_filename=args.code_filename,
        num_parts=args.num_parts,
        num_k_per_part=args.num_k_per_part,
        use_gt_label=args.use_gt_label,
        bg_code=args.bg_code,
        repeat=args.repeat,
        extra_data_dir=args.extra_data_dir,
        shape_attn_loss=args.shape_attn_loss,
    )

    with accelerator.main_process_first():
        if args.filter_class is not None:
            train_dataset.filter_by_class(args.filter_class)
        if args.max_train_samples is not None:
            train_dataset.set_max_samples(args.max_train_samples, args.seed)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn(args, tokenizer, placeholder_token),
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )
    simple_mapper = accelerator.prepare(simple_mapper)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2image-fine-tune", config=vars(args))

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
    )
    progress_bar.set_description("Steps")

    print(simple_mapper)

    placeholder_token_ids_for_loss = placeholder_token_ids
    if args.learn_whole:
        # skip the global token in attention loss
        placeholder_token_ids_for_loss = placeholder_token_ids[:-1]

    with open(args.output_dir + "/config.yaml", "w+") as f:
        yaml.dump(vars(args), f)

    epoch = 0

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        train_attn_loss = 0.0
        train_diff_loss = 0.0
        train_kl_loss = 0.0
        train_mixed_reg_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet, simple_mapper):
                bsz = batch["pixel_values"].shape[0]

                # Convert images to latent space
                vae_input = batch["pixel_values"].to(dtype=weight_dtype)

                latents = vae.encode(vae_input).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=accelerator.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if args.mixed_latent_reg != 0:
                    noise_a = torch.randn_like(noise)
                    noise_b = torch.randn_like(noise)
                    noisy_latents_a = noise_scheduler.add_noise(
                        latents, noise_a, timesteps
                    )
                    noisy_latents_b = noise_scheduler.add_noise(
                        latents, noise_b, timesteps
                    )

                obtain_segmentation_mask_(
                    batch,
                    dino,
                    seg,
                    accelerator,
                    args.shape_attn_loss,
                    timesteps.cpu().tolist(),
                    args.t_threshold,
                    args.recaption,
                    args.drop_tokens,
                    args.drop_rate,
                    tokenizer,
                    args.vector_shuffle,
                    args.use_templates,
                    placeholder_token,
                    args.skip_background,
                    args.shared_shape_token,
                    args.num_k_per_part,
                    keep_partids,
                    seg.background_code if not args.remove_unused else 0,
                    args.enable_cfg,
                    args.learn_whole,
                )

                if args.part_craft_token_mapper:
                    mapper_outputs = simple_mapper(batch["codes"])
                else:
                    sampled, mu, logvar = simple_mapper.forward_dino(
                        batch["dino_cls"],
                        batch["codes"],
                        update=True,
                        ema_rate=args.ema_rate,
                    )
                    mapper_outputs = simple_mapper(sampled, input_is_mu=True)

                # Get the text embedding for conditioning
                modified_hs = text_encoder.text_model.forward_embeddings_with_mapper(
                    batch["input_ids"], None, mapper_outputs, placeholder_token_ids
                )
                encoder_hidden_states = text_encoder(
                    batch["input_ids"], hidden_states=modified_hs
                )[0]

                #### get a/b condition ####
                with torch.no_grad():
                    random_mu = simple_mapper.random_sample_mu(
                        bsz, use_dino_features=True
                    )

                with torch.set_grad_enabled(args.enable_grad_for_mu):
                    random_mapper_outputs = simple_mapper(random_mu, input_is_mu=True)
                    random_modified_hs = (
                        text_encoder.text_model.forward_embeddings_with_mapper(
                            batch["input_ids"],
                            None,
                            random_mapper_outputs,
                            placeholder_token_ids,
                        )
                    )
                    # print(modified_hs.size())
                    random_encoder_hidden_states = text_encoder(
                        batch["input_ids"], hidden_states=random_modified_hs
                    )[0]
                ########

                # Get the target for loss depending on the prediction type
                if args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=args.prediction_type
                    )

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample

                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    attn_loss, max_attn = dreamcreature_loss_v2(
                        batch,
                        unet,
                        placeholder_token_ids_for_loss,
                        accelerator,
                        attn_size,
                        args.selfattn_loss,
                        args.selfattn_loss_method,
                        args.shape_attn_loss,
                        args.background_attn_loss_scale,
                        not args.no_attn_norm,
                        args.attn_scale,
                        args.exp_attn,
                        args.gs_blur_attn,
                        store_embeddings=True,
                    )

                    if args.masked_training:
                        masks = batch["image_masks"].unsqueeze(1).to(accelerator.device)
                        loss_image_mask = F.interpolate(
                            masks.float(), size=target.shape[-2:], mode="bilinear"
                        ) * torch.ones_like(target)
                        loss = loss * loss_image_mask
                        loss = loss.sum() / loss_image_mask.sum()
                    else:
                        loss = loss.mean()
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr
                    )

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    attn_loss, max_attn = dreamcreature_loss_v2(
                        batch,
                        unet,
                        placeholder_token_ids_for_loss,
                        accelerator,
                        attn_size,
                        args.selfattn_loss,
                        args.selfattn_loss_method,
                        args.shape_attn_loss,
                        args.background_attn_loss_scale,
                        not args.no_attn_norm,
                        args.attn_scale,
                        args.exp_attn,
                        args.gs_blur_attn,
                        store_embeddings=True,
                    )

                    if args.masked_training:
                        masks = batch["image_masks"].unsqueeze(1).to(accelerator.device)
                        loss_image_mask = F.interpolate(
                            masks.float(), size=target.shape[-2:], mode="bilinear"
                        ) * torch.ones_like(target)
                        loss = loss * loss_image_mask
                        loss = (
                            loss.sum(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.sum() / loss_image_mask.sum()
                    else:
                        loss = (
                            loss.mean(dim=list(range(1, len(loss.shape))))
                            * mse_loss_weights
                        )
                        loss = loss.mean()

                diff_loss = loss.clone().detach()
                avg_diff_loss = accelerator.gather(
                    diff_loss.repeat(args.train_batch_size)
                ).mean()
                train_diff_loss += (
                    avg_diff_loss.item() / args.gradient_accumulation_steps
                )

                #### part reg loss ####
                if args.kl_loss != 0:
                    if logvar is None:
                        embeds_mu = mu
                        kl_loss = embeds_mu.pow(2).mean()
                    else:
                        embeds_mu = mu
                        embeds_logvar = logvar

                        kl_loss = -0.5 * torch.sum(
                            1 + embeds_logvar - embeds_mu.pow(2) - embeds_logvar.exp()
                        )
                        kl_loss /= args.num_parts
                        kl_loss = kl_loss.mean()

                    loss += args.kl_loss * kl_loss

                    avg_kl_loss = accelerator.gather(
                        kl_loss.repeat(args.train_batch_size)
                    ).mean()
                    train_kl_loss += (
                        avg_kl_loss.item() / args.gradient_accumulation_steps
                    )
                else:
                    kl_loss = torch.tensor(0.0)

                if args.mixed_latent_reg != 0:
                    #### a ####
                    model_pred_a = unet(
                        noisy_latents_a, timesteps, random_encoder_hidden_states
                    ).sample
                    model_pred_a_attn_loss, _ = dreamcreature_loss_v2(
                        batch,
                        unet,
                        placeholder_token_ids_for_loss,
                        accelerator,
                        attn_size,
                        args.selfattn_loss,
                        args.selfattn_loss_method,
                        args.shape_attn_loss,
                        args.background_attn_loss_scale,
                        not args.no_attn_norm,
                        args.attn_scale,
                        args.exp_attn,
                        args.gs_blur_attn,
                        store_embeddings=True,
                    )
                    model_pred_a_embeddings = obtain_embeddings(
                        batch, args.mode, args.topk
                    )

                    #### b ####
                    model_pred_b = unet(
                        noisy_latents_b, timesteps, random_encoder_hidden_states
                    ).sample
                    model_pred_b_attn_loss, _ = dreamcreature_loss_v2(
                        batch,
                        unet,
                        placeholder_token_ids_for_loss,
                        accelerator,
                        attn_size,
                        args.selfattn_loss,
                        args.selfattn_loss_method,
                        args.shape_attn_loss,
                        args.background_attn_loss_scale,
                        not args.no_attn_norm,
                        args.attn_scale,
                        args.exp_attn,
                        args.gs_blur_attn,
                        store_embeddings=True,
                    )
                    model_pred_b_embeddings = obtain_embeddings(
                        batch, args.mode, args.topk
                    )

                    mixed_reg_loss = 0.0
                    for a in model_pred_a_embeddings.keys():
                        spatial_loss = F.mse_loss(
                            model_pred_a_embeddings[a], model_pred_b_embeddings[a]
                        )

                        mixed_reg_loss += spatial_loss

                    if args.add_reg_attn:
                        attn_loss += (
                            model_pred_a_attn_loss + model_pred_b_attn_loss
                        ) * 0.5

                    loss += args.mixed_latent_reg * mixed_reg_loss

                    avg_mixed_reg_loss = accelerator.gather(
                        mixed_reg_loss.repeat(args.train_batch_size)
                    ).mean()
                    train_mixed_reg_loss += (
                        avg_mixed_reg_loss.item() / args.gradient_accumulation_steps
                    )
                else:
                    mixed_reg_loss = torch.tensor(0.0)

                avg_attn_loss = accelerator.gather(
                    attn_loss.repeat(args.train_batch_size)
                ).mean()
                train_attn_loss += (
                    avg_attn_loss.item() / args.gradient_accumulation_steps
                )

                loss += args.attn_loss * attn_loss

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = list(lora_layers.parameters()) + list(
                        simple_mapper.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if isinstance(max_attn, (tuple, list)):
                    max_attn, max_selfattn = max_attn

                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "diff_loss": train_diff_loss,
                            "attn_loss": train_attn_loss,
                            "kl_loss": train_kl_loss,
                            "mixed_reg_loss": train_mixed_reg_loss,
                            "mapper_norm": mapper_outputs.detach().norm().item(),
                            "max_attn": max_attn.item(),
                            "max_selfattn": max_selfattn.item(),
                        },
                        step=global_step,
                    )
                else:
                    accelerator.log(
                        {
                            "train_loss": train_loss,
                            "diff_loss": train_diff_loss,
                            "attn_loss": train_attn_loss,
                            "kl_loss": train_kl_loss,
                            "mixed_reg_loss": train_mixed_reg_loss,
                            "mapper_norm": mapper_outputs.detach().norm().item(),
                            "max_attn": max_attn.item(),
                        },
                        step=global_step,
                    )
                train_loss = 0.0
                train_attn_loss = 0.0
                train_diff_loss = 0.0
                train_kl_loss = 0.0
                train_mixed_reg_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": diff_loss.detach().item(),
                "attn_loss": attn_loss.detach().item(),
                "kl_loss": kl_loss.detach().item(),
                "mixed_reg_loss": mixed_reg_loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            if (
                args.validation_prompt is not None
                and epoch % args.validation_epochs == 0
            ):
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )

                scheduler_kwargs = {}
                if args.prediction_type == "v_prediction":
                    scheduler_kwargs = {
                        "rescale_betas_zero_snr": True,
                        "timestep_spacing": "trailing",
                    }

                scheduler = DDIMScheduler.from_pretrained(
                    args.pretrained_model_name_or_path,
                    subfolder="scheduler",
                    **scheduler_kwargs,
                )
                if args.prediction_type is not None:
                    scheduler.register_to_config(prediction_type=args.prediction_type)

                pipeline = DreamCreatureSDPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    unet=accelerator.unwrap_model(unet),
                    scheduler=scheduler,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
                pipeline.placeholder_token_ids = placeholder_token_ids
                pipeline.simple_mapper = accelerator.unwrap_model(simple_mapper)
                pipeline.replace_token = False
                pipeline.v = "re"
                # pipeline.verbose = True

                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device)
                if args.seed is not None:
                    generator = generator.manual_seed(args.seed)
                images = []
                for ni in range(args.num_validation_images):
                    # images.append(
                    #     pipeline(args.validation_prompt, num_inference_steps=30, generator=generator,
                    #              height=args.resolution, width=args.resolution).images[0]
                    # )
                    # generator = generator.manual_seed(args.seed + args.seed * ni)
                    camera = (
                        get_camera(4, elevation=15, azimuth_start=0)
                        .to(accelerator.device)
                        .to(weight_dtype)
                    )
                    images.extend(
                        pipeline(
                            args.validation_prompt,
                            num_inference_steps=30,
                            generator=generator,
                            height=256,
                            width=256,
                            camera=camera,
                            guidance_scale=args.gs,
                            negative_prompt=args.negative_prompt,
                        ).images
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images(
                            "validation", np_images, epoch, dataformats="NHWC"
                        )
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(
                                        image, caption=f"{i}: {args.validation_prompt}"
                                    )
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # unet = unet.to(torch.float32)
        # unet.save_attn_procs(args.output_dir, safe_serialization=not args.custom_diffusion)

        torch.save(
            lora_layers.to(torch.float32).state_dict(),
            args.output_dir + "/lora_layers.pth",
        )
        torch.save(
            simple_mapper.to(torch.float32).state_dict(),
            args.output_dir + "/hash_mapper.pth",
        )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                dataset_name=args.dataset_name,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        del unet

    # Final inference
    # Load previous pipeline
    tokenizer = MultiTokenCLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    text_encoder = CustomCLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
    )
    unet: MVDreamUNet2DConditionModel = get_mvdream_unet()

    scheduler_kwargs = {}
    if args.prediction_type == "v_prediction":
        scheduler_kwargs = {
            "rescale_betas_zero_snr": True,
            "timestep_spacing": "trailing",
        }

    scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        **scheduler_kwargs,
    )
    if args.prediction_type is not None:
        scheduler.register_to_config(prediction_type=args.prediction_type)

    pipeline = DreamCreatureSDPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        unet=unet,
        scheduler=scheduler,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    placeholder_token = "<part>"
    initializer_token = None
    additional = 1 if args.learn_whole else 0
    placeholder_token_ids = add_tokens(
        tokenizer,
        text_encoder,
        placeholder_token,
        args.num_parts + additional,
        initializer_token,
    )
    pipeline.placeholder_token_ids = placeholder_token_ids
    pipeline.simple_mapper = setup_token_mapper(args, OUT_DIMS)
    pipeline.simple_mapper.load_state_dict(
        torch.load(args.output_dir + "/hash_mapper.pth", map_location="cpu")
    )
    pipeline.simple_mapper.to(accelerator.device)

    pipeline = pipeline.to(accelerator.device)

    # load attention processors
    # pipeline.unet.load_attn_procs(args.output_dir, use_safetensors=not args.custom_diffusion)
    setup_attn_processor(
        pipeline.unet,
        rank=args.rank,
        attn_size=attn_size,
        resolution=args.resolution,
        selfattn_loss=args.selfattn_loss,
        tag_embeddings=False,
        skip_sa=args.skip_sa,
    )
    load_attn_processor(pipeline.unet, args.output_dir + "/lora_layers.pth")

    pipeline = pipeline.to(weight_dtype)

    # run inference
    pipeline.replace_token = False
    pipeline.v = "re"

    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    # images = []
    # for _ in range(args.num_validation_images):
    #     images.append(pipeline(args.validation_prompt, num_inference_steps=30, generator=generator,
    #                            height=args.resolution, width=args.resolution).images[0])

    images = []
    for ni in range(args.num_validation_images // 4):
        # images.append(
        #     pipeline(args.validation_prompt, num_inference_steps=30, generator=generator,
        #              height=args.resolution, width=args.resolution).images[0]
        # )
        # generator = generator.manual_seed(args.seed + args.seed * ni)
        camera = (
            get_camera(4, elevation=15, azimuth_start=0)
            .to(accelerator.device)
            .to(weight_dtype)
        )
        images.extend(
            pipeline(
                args.validation_prompt,
                num_inference_steps=30,
                generator=generator,
                height=256,
                width=256,
                camera=camera,
                guidance_scale=args.gs,
                negative_prompt=args.negative_prompt,
            ).images
        )

    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if len(images) != 0:
                if tracker.name == "tensorboard":
                    np_images = np.stack([np.asarray(img) for img in images])
                    tracker.writer.add_images(
                        "test", np_images, epoch, dataformats="NHWC"
                    )
                if tracker.name == "wandb":
                    tracker.log(
                        {
                            "test": [
                                wandb.Image(
                                    image, caption=f"{i}: {args.validation_prompt}"
                                )
                                for i, image in enumerate(images)
                            ]
                        }
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
