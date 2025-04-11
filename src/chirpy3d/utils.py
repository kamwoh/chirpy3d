import torch
import torch.nn as nn
# from accelerate.logging import get_logger
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from partcraft.attn_processor import LoRAAttnProcessorCustom
from partcraft.mapper import TokenMapper, PartCraftTokenMapper


# logger = get_logger(__name__, log_level="INFO")


def add_tokens(
    tokenizer,
    text_encoder,
    placeholder_token,
    num_vec_per_token=1,
    initializer_token=None,
):
    """
    Add tokens to the tokenizer and set the initial value of token embeddings
    """
    tokenizer.add_placeholder_tokens(
        placeholder_token, num_vec_per_token=num_vec_per_token
    )
    text_encoder.resize_token_embeddings(len(tokenizer))
    token_embeds = text_encoder.get_input_embeddings().weight.data
    placeholder_token_ids = tokenizer.encode(
        placeholder_token, add_special_tokens=False
    )
    if initializer_token:
        token_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = token_embeds[
                token_ids[i * len(token_ids) // num_vec_per_token]
            ]
    else:
        for i, placeholder_token_id in enumerate(placeholder_token_ids):
            token_embeds[placeholder_token_id] = torch.randn_like(
                token_embeds[placeholder_token_id]
            )
    return placeholder_token_ids


def tokenize_prompt(tokenizer, prompt, replace_token=False):
    text_inputs = tokenizer(
        prompt,
        replace_token=replace_token,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def get_processor(self, return_deprecated_lora: bool = False):
    r"""
    Get the attention processor in use.

    Args:
        return_deprecated_lora (`bool`, *optional*, defaults to `False`):
            Set to `True` to return the deprecated LoRA attention processor.

    Returns:
        "AttentionProcessor": The attention processor in use.
    """
    if not return_deprecated_lora:
        return self.processor

    # TODO(Sayak, Patrick). The rest of the function is needed to ensure backwards compatible
    # serialization format for LoRA Attention Processors. It should be deleted once the integration
    # with PEFT is completed.
    is_lora_activated = {
        name: module.lora_layer is not None
        for name, module in self.named_modules()
        if hasattr(module, "lora_layer")
    }

    # 1. if no layer has a LoRA activated we can return the processor as usual
    if not any(is_lora_activated.values()):
        return self.processor

    # If doesn't apply LoRA do `add_k_proj` or `add_v_proj`
    is_lora_activated.pop("add_k_proj", None)
    is_lora_activated.pop("add_v_proj", None)
    # 2. else it is not posssible that only some layers have LoRA activated
    if not all(is_lora_activated.values()):
        raise ValueError(
            f"Make sure that either all layers or no layers have LoRA activated, but have {is_lora_activated}"
        )

    # 3. And we need to merge the current LoRA layers into the corresponding LoRA attention processor
    # non_lora_processor_cls_name = self.processor.__class__.__name__
    # lora_processor_cls = getattr(import_module(__name__), "LoRA" + non_lora_processor_cls_name)

    hidden_size = self.inner_dim

    # now create a LoRA attention processor from the LoRA layers
    kwargs = {
        "cross_attention_dim": self.cross_attention_dim,
        "rank": self.to_q.lora_layer.rank,
        "network_alpha": self.to_q.lora_layer.network_alpha,
        "q_rank": self.to_q.lora_layer.rank,
        "q_hidden_size": self.to_q.lora_layer.out_features,
        "k_rank": self.to_k.lora_layer.rank,
        "k_hidden_size": self.to_k.lora_layer.out_features,
        "v_rank": self.to_v.lora_layer.rank,
        "v_hidden_size": self.to_v.lora_layer.out_features,
        "out_rank": self.to_out[0].lora_layer.rank,
        "out_hidden_size": self.to_out[0].lora_layer.out_features,
    }

    if hasattr(self.processor, "attention_op"):
        kwargs["attention_op"] = self.processor.attention_op

    lora_processor = LoRAAttnProcessor(hidden_size, **kwargs)
    lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
    lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
    lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
    lora_processor.to_out_lora.load_state_dict(self.to_out[0].lora_layer.state_dict())

    return lora_processor


def get_attn_processors(self):
    r"""
    Returns:
        `dict` of attention processors: A dictionary containing all attention processors used in the model with
        indexed by its weight name.
    """
    # set recursively
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if hasattr(module, "get_processor"):
            processors[f"{name}.processor"] = get_processor(
                module, return_deprecated_lora=True
            )

        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

        return processors

    for name, module in self.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


def setup_attn_processor(unet, **kwargs):
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]

        lora_attn_procs[name] = LoRAAttnProcessorCustom(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=kwargs["rank"],
            attn_size=(
                kwargs["attn_size"]
                if kwargs["attn_size"] is not None
                else kwargs["resolution"] // 32
            ),
            tag_self_attn=kwargs["selfattn_loss"],
            tag_embeddings=kwargs.get("tag_embeddings", False),
            skip_sa="attn1" in name and kwargs.get("skip_sa", False),
            init_lora_weights=kwargs.get("init_lora_weights", False),
        )

    unet.set_attn_processor(lora_attn_procs)


def load_attn_processor(unet, filename):
    # logger.info(f'Load attn processors from {filename}')
    print(f"Load attn processors from {filename}")
    lora_layers = AttnProcsLayers(get_attn_processors(unet))
    if "safetensors" in filename:
        from safetensors.torch import load_file

        lora_layers.load_state_dict(load_file(filename))
    else:
        lora_layers.load_state_dict(torch.load(filename, map_location="cpu"))


def setup_token_mapper(args, OUT_DIMS):
    class Sine(nn.Module):
        def __init__(self, w=1.0):
            super().__init__()

            self.w = w

        def forward(self, x):
            return torch.sin(x * self.w)

    act_fn = {"relu": nn.ReLU(), "silu": nn.SiLU(), "sine": Sine(w=1.0)}[
        args.projection_actfn
    ]

    if args.part_craft_token_mapper:
        mapper = PartCraftTokenMapper(
            args.num_parts,
            args.num_k_per_part,
            OUT_DIMS,
            args.projection_nlayers,
            act_fn,
            not args.no_pe,
        )
    else:
        if hasattr(args, "learn_whole"):
            additional = 1 if args.learn_whole else 0
        else:
            additional = 0

        mapper = TokenMapper(
            args.num_parts + additional,
            args.num_k_per_part,
            OUT_DIMS,
            args.projection_nlayers,
            act_fn,
            vae_dims=args.vae_dims,
            concat_pe=args.concat_pe,
            learnable_dino_embeddings=args.learnable_dino_embeddings,
            dino_nlayers=args.dino_nlayers,
        )
    return mapper


def load_pipeline():
    # TODO: implement this!!
    pass
