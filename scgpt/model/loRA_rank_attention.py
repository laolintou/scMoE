import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from torch.nn.modules.activation import _is_make_fx_tracing, _check_arg_device, _arg_requires_grad
from torch.nn.parameter import Parameter
from torch.nn import functional as F



def apply_rotary_emb(q_pe, freqs_cis):
    # Placeholder for the actual rotary embedding application logic
    return q_pe * freqs_cis
from torch import Tensor
class MLA(nn.Module):
    """
    Modified Multi-Headed Attention Layer (MLA) with Low-Rank Projection.

    Attributes:
        embed_dim (int): Dimensionality of the input features.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability applied to the attention weights.
        bias (bool): If set to False, layers will not learn an additive bias.
        add_bias_kv (bool): If set to True, biases are added to keys and values.
        add_zero_attn (bool): If set to True, an extra “zero” attention is added to the scaled dot-product attention.
        kdim (int): Total number of features in key. Default: embed_dim.
        vdim (int): Total number of features in value. Default: embed_dim.
        batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature).
        device (torch.device): Device on which the model parameters will be allocated.
        dtype (torch.dtype): Desired data type of the model parameters.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
    """
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 qkv_rank=None, out_rank=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # 设置默认的秩
        if qkv_rank is None:
            qkv_rank = max(1, embed_dim // 4)  # 示例默认值
        if out_rank is None:
            out_rank = max(1, embed_dim // 4)
        self.qkv_rank = qkv_rank
        self.out_rank = out_rank

        if not self._qkv_same_embed_dim:
            # 分解Q、K、V的投影矩阵
            self.q_proj_A = Parameter(torch.empty((embed_dim, qkv_rank), **factory_kwargs))
            self.q_proj_B = Parameter(torch.empty((qkv_rank, embed_dim), **factory_kwargs))
            self.k_proj_A = Parameter(torch.empty((embed_dim, qkv_rank), **factory_kwargs))
            self.k_proj_B = Parameter(torch.empty((qkv_rank, self.kdim), **factory_kwargs))
            self.v_proj_A = Parameter(torch.empty((embed_dim, qkv_rank), **factory_kwargs))
            self.v_proj_B = Parameter(torch.empty((qkv_rank, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)
        else:
            # 分解in_proj_weight为低秩矩阵
            self.in_proj_A = Parameter(torch.empty((3 * embed_dim, qkv_rank), **factory_kwargs))
            self.in_proj_B = Parameter(torch.empty((qkv_rank, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)

        # 分解输出投影层
        self.out_proj_A = Parameter(torch.empty((embed_dim, out_rank), **factory_kwargs))
        self.out_proj_B = Parameter(torch.empty((out_rank, embed_dim), **factory_kwargs))
        if bias:
            self.out_proj_bias = Parameter(torch.empty(embed_dim, **factory_kwargs))
        else:
            self.register_parameter('out_proj_bias', None)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_A)
            nn.init.xavier_uniform_(self.in_proj_B)
        else:
            nn.init.xavier_uniform_(self.q_proj_A)
            nn.init.xavier_uniform_(self.q_proj_B)
            nn.init.xavier_uniform_(self.k_proj_A)
            nn.init.xavier_uniform_(self.k_proj_B)
            nn.init.xavier_uniform_(self.v_proj_A)
            nn.init.xavier_uniform_(self.v_proj_B)

        nn.init.xavier_uniform_(self.out_proj_A)
        nn.init.xavier_uniform_(self.out_proj_B)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        if self.out_proj_bias is not None:
            nn.init.constant_(self.out_proj_bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

        # if not is_fastpath_enabled:
        #     why_not_fast_path = "torch.backends.mha.get_fastpath_enabled() was not True"
        # elif not is_batched:
        #     why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        # elif query is not key or key is not value:
        #     # When lifting this restriction, don't forget to either
        #     # enforce that the dtypes all match or test cases where
        #     # they don't!
        #     why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        # elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
        #     why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        # elif self.in_proj_weight is None:
        #     why_not_fast_path = "in_proj_weight was None"
        # elif query.dtype != self.in_proj_weight.dtype:
        #     # this case will fail anyway, but at least they'll get a useful error message.
        #     why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        # elif self.training:
        #     why_not_fast_path = "training is enabled"
        # elif (self.num_heads % 2) != 0:
        #     why_not_fast_path = "self.num_heads is not even"
        # elif not self.batch_first:
        #     why_not_fast_path = "batch_first was not True"
        # elif self.bias_k is not None:
        #     why_not_fast_path = "self.bias_k was not None"
        # elif self.bias_v is not None:
        #     why_not_fast_path = "self.bias_v was not None"
        # elif self.add_zero_attn:
        #     why_not_fast_path = "add_zero_attn was enabled"
        # elif not self._qkv_same_embed_dim:
        #     why_not_fast_path = "_qkv_same_embed_dim was not True"
        # elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
        #     why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
        #                          is not supported with NestedTensor input"
        # elif torch.is_autocast_enabled():
        #     why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # 生成低秩权重
        if self._qkv_same_embed_dim:
            in_proj_weight = torch.mm(self.in_proj_A, self.in_proj_B)
        else:
            q_proj_weight = torch.mm(self.q_proj_A, self.q_proj_B)
            k_proj_weight = torch.mm(self.k_proj_A, self.k_proj_B)
            v_proj_weight = torch.mm(self.v_proj_A, self.v_proj_B)

        out_proj_weight = torch.mm(self.out_proj_A, self.out_proj_B)

        # 调用F.multi_head_attention_forward
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                None, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, self.out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight,
                k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj_weight, self.out_proj_bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)

        # 调整batch_first的输出
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


