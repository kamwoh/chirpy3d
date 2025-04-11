import math

import einops
from diffusers.models.attention_processor import *


class LoRAAttnProcessorCustom(AttnProcessor, nn.Module):
    r"""
    Processor for implementing the LoRA attention mechanism.

    Args:
        hidden_size (`int`, *optional*):
            The hidden size of the attention layer.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the `encoder_hidden_states`.
        rank (`int`, defaults to 4):
            The dimension of the LoRA update matrices.
        network_alpha (`int`, *optional*):
            Equivalent to `alpha` but it's usage is specific to Kohya (A1111) style LoRAs.
    """

    def __init__(
        self,
        hidden_size,
        cross_attention_dim=None,
        rank=4,
        network_alpha=None,
        init_lora_weights=False,
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attn_size = kwargs.pop("attn_size", 16)
        self.qk_only = kwargs.pop("qk_only", False)
        self.q_only = kwargs.pop("q_only", False)
        self.tag_self_attn = kwargs.pop("tag_self_attn", False)
        self.skip_sa = kwargs.pop("skip_sa", False)
        self.tag_embeddings = kwargs.pop("tag_embeddings", False)

        q_rank = kwargs.pop("q_rank", None)
        q_hidden_size = kwargs.pop("q_hidden_size", None)
        q_rank = q_rank if q_rank is not None else rank
        q_hidden_size = q_hidden_size if q_hidden_size is not None else hidden_size

        v_rank = kwargs.pop("v_rank", None)
        v_hidden_size = kwargs.pop("v_hidden_size", None)
        v_rank = v_rank if v_rank is not None else rank
        v_hidden_size = v_hidden_size if v_hidden_size is not None else hidden_size

        out_rank = kwargs.pop("out_rank", None)
        out_hidden_size = kwargs.pop("out_hidden_size", None)
        out_rank = out_rank if out_rank is not None else rank
        out_hidden_size = (
            out_hidden_size if out_hidden_size is not None else hidden_size
        )

        if not self.skip_sa:
            self.to_q_lora = LoRALinearLayer(
                q_hidden_size, q_hidden_size, q_rank, network_alpha
            )
            self.init_default_(self.to_q_lora, init_lora_weights)
            if not self.q_only:
                self.to_k_lora = LoRALinearLayer(
                    cross_attention_dim or hidden_size, hidden_size, rank, network_alpha
                )
                self.init_default_(self.to_k_lora, init_lora_weights)
                if not self.qk_only:
                    self.to_v_lora = LoRALinearLayer(
                        cross_attention_dim or v_hidden_size,
                        v_hidden_size,
                        v_rank,
                        network_alpha,
                    )
                    self.to_out_lora = LoRALinearLayer(
                        out_hidden_size, out_hidden_size, out_rank, network_alpha
                    )
                    self.init_default_(self.to_v_lora, init_lora_weights)
                    self.init_default_(self.to_out_lora, init_lora_weights)

    def init_default_(self, layer, init_lora_weights=False):
        if init_lora_weights:
            nn.init.kaiming_uniform_(layer.down.weight, a=math.sqrt(5))

    def __call__(self, attn: Attention, hidden_states, *args, **kwargs):
        self_cls_name = self.__class__.__name__
        # deprecate(
        #     self_cls_name,
        #     "0.26.0",
        #     (
        #         f"Make sure use {self_cls_name[4:]} instead by setting"
        #         "LoRA layers to `self.{to_q,to_k,to_v,to_out[0]}.lora_layer` respectively. This will be done automatically when using"
        #         " `LoraLoaderMixin.load_lora_weights`"
        #     ),
        # )

        if not self.skip_sa:
            attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
            if not self.q_only:
                attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
                if not self.qk_only:
                    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
                    attn.to_out[0].lora_layer = self.to_out_lora.to(
                        hidden_states.device
                    )

        attn._modules.pop("processor")
        attn.processor = AttnProcessorCustom(
            self.attn_size, self.tag_self_attn, self.tag_embeddings
        )
        return attn.processor(attn, hidden_states, *args, **kwargs)


class AttnProcessorCustom(AttnProcessor):
    r"""
    Default processor for performing attention-related computations.
    """

    def __init__(self, attn_size, tag_self_attn=False, tag_embeddings=False):
        self.attn_size = attn_size
        self.tag_self_attn = tag_self_attn
        self.tag_embeddings = tag_embeddings

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
    ):
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        if hasattr(attn, "attn_size"):
            attn_size = attn.attn_size
        else:
            attn_size = self.attn_size

        if isinstance(attn_size, int):
            attn_size = [attn_size]

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        total_qk = attention_probs.size(1)
        sqrt_total = int(total_qk**0.5)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        attn.embeddings = None

        if self.tag_embeddings == "all":
            attn.embeddings = hidden_states.clone()

        if attention_probs.size(2) == 77 and sqrt_total in attn_size:  # (B*Head,HW,L)
            attn_probs_cache = attention_probs.reshape(
                batch_size, -1, sqrt_total, sqrt_total, 77
            )
            attn.attn_probs = attn_probs_cache
            attn.self_attn_probs = None
            if self.tag_embeddings:
                attn.embeddings = hidden_states.clone()  # (B,C,H,W)
        elif (
            attention_probs.size(2) != 77
            and sqrt_total in attn_size
            and self.tag_self_attn
        ):
            attn_probs_cache = attention_probs.reshape(
                batch_size, -1, sqrt_total * sqrt_total, sqrt_total * sqrt_total
            )
            attn.self_attn_probs = attn_probs_cache
            attn.attn_probs = None
            if self.tag_embeddings:
                attn.embeddings = hidden_states.clone()
        else:
            attn.attn_probs = None
            attn.self_attn_probs = None

        return hidden_states


def cape_embed(f, P):
    # f is feature vector of shape [..., d]
    # P is 4x4 transformation matrix
    f = einops.rearrange(f, "... (d k) -> ... d k", k=4)
    return einops.rearrange(f @ P, "... d k -> ... (d k)", k=4)


class CAPEAttentionProcessor(AttnProcessor):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        posemb: Optional[torch.FloatTensor] = None,  # passed by cross_attention_kwargs
    ) -> torch.Tensor:
        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        if posemb is not None:
            # turn 2d attention into multiview attention
            self_attn = (
                encoder_hidden_states is None
            )  # check if self attn or cross attn
            [p_out, p_out_inv], [p_in, p_in_inv] = posemb
            t_out, t_in = p_out.shape[1], p_in.shape[1]  # t size
            hidden_states = einops.rearrange(
                hidden_states, "(b t_out) l d -> b (t_out l) d", t_out=t_out
            )

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(
                1, 2
            )

        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(
                encoder_hidden_states
            )

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if posemb is not None:
            p_out_inv = einops.repeat(
                p_out_inv, "b t_out f g -> b (t_out l) f g", l=query.shape[1] // t_out
            )  # query shape
            if self_attn:
                p_in = einops.repeat(
                    p_out, "b t_out f g -> b (t_out l) f g", l=query.shape[1] // t_out
                )  # query shape
            else:
                p_in = einops.repeat(
                    p_in, "b t_in f g -> b (t_in l) f g", l=key.shape[1] // t_in
                )  # key shape

            # query f_q @ (p_out)^(-T) .permute(0, 1, 3, 2)
            query = cape_embed(query, p_out_inv)
            key = cape_embed(key, p_in)  # key f_k @ p_in

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if posemb is not None:
            # reshape back
            hidden_states = einops.rearrange(
                hidden_states, "b (t_out l) d -> (b t_out) l d", t_out=t_out
            )

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
