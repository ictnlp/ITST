# Speech-to-text simultaneous translation based on information-transport-based policy
#
# This source code is from Shaolei Zhang


import math
import pdb
import torch
from torch import Tensor
import torch.nn as nn
from fairseq import utils

from examples.simultaneous_translation.utils.p_choose_strategy import (
    learnable_p_choose,
    waitk_p_choose,
)

from examples.simultaneous_translation.utils.monotonic_attention import (
    expected_alignment_from_p_choose,
    expected_soft_attention,
    mass_preservation,
)

from examples.simultaneous_translation.utils.truncated_attention import (
    truncated_soft_attention,
    rw_with_info,
)

from fairseq.modules import MultiheadAttention

from . import register_monotonic_attention
from typing import Dict, Optional


@register_monotonic_attention("ITST")
class InformationTransportAttention(MultiheadAttention):
    """
    Abstract class of monotonic attentions
    """

    k_in_proj: Dict[str, nn.Linear]
    q_in_proj: Dict[str, nn.Linear]

    def __init__(self, args):
        super().__init__(
            embed_dim=args.decoder_embed_dim,
            num_heads=args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.soft_attention = True

        self.num_rw = 1

        self.eps = getattr(args, "attention_eps", True)
        self.mass_preservation = getattr(args, "mass_preservation", True)

        self.noise_type = args.noise_type
        self.noise_mean = args.noise_mean
        self.noise_var = args.noise_var

        self.energy_bias_init = args.energy_bias_init
        self.energy_bias = (
            nn.Parameter(self.energy_bias_init * torch.ones([1]))
            if args.energy_bias is True
            else 0
        )

        self.k_in_proj = {"transport": self.k_proj}
        self.q_in_proj = {"transport": self.q_proj}
        self.init_soft_attention()
        self.chunk_size = None

        self.test_threshold = getattr(args, "test_threshold", 10)

        no_fuse_ot_with_attn = getattr(args, "no_fuse_ot_with_attn", False)
        self.fuse_ot_with_attn = not no_fuse_ot_with_attn

    def init_soft_attention(self):
        self.k_proj_soft = nn.Linear(self.kdim, self.embed_dim, bias=True)
        self.q_proj_soft = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.k_in_proj["soft"] = self.k_proj_soft
        self.q_in_proj["soft"] = self.q_proj_soft

        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(
                self.k_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
            nn.init.xavier_uniform_(
                self.q_in_proj["soft"].weight, gain=1 / math.sqrt(2)
            )
        else:
            nn.init.xavier_uniform_(self.k_in_proj["soft"].weight)
            nn.init.xavier_uniform_(self.q_in_proj["soft"].weight)

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--no-mass-preservation', action="store_false",
                            dest="mass_preservation",
                            help='Do not stay on the last token when decoding')
        parser.add_argument('--mass-preservation', action="store_true",
                            dest="mass_preservation",
                            help='Stay on the last token when decoding')
        parser.set_defaults(mass_preservation=True)
        parser.add_argument('--noise-var', type=float, default=1.0,
                            help='Variance of discretness noise')
        parser.add_argument('--noise-mean', type=float, default=0.0,
                            help='Mean of discretness noise')
        parser.add_argument('--noise-type', type=str, default="flat",
                            help='Type of discretness noise')
        parser.add_argument('--energy-bias', action="store_true",
                            default=False,
                            help='Bias for energy')
        parser.add_argument('--energy-bias-init', type=float, default=-2.0,
                            help='Initial value of the bias for energy')
        parser.add_argument('--attention-eps', type=float, default=1e-6,
                            help='Epsilon when calculating expected attention')
        parser.add_argument('--test-threshold', type=float, default=10.0,
                            help="if > test_threshold, start translating.")
        parser.add_argument('--no-fuse-ot-with-attn', action="store_true", default=False,help="dont multiple ot matrix with attention")

    def energy_from_qk(
        self,
        query: Tensor,
        key: Tensor,
        energy_type: str,
        key_padding_mask: Optional[Tensor] = None,
        bias: int = 0,
    ):
        """
        Compute energy from query and key
        q_func_value is a tuple looks like
        (q_proj_func, q_tensor)
        q_tensor size: bsz, tgt_len, emb_dim
        k_tensor size: bsz, src_len, emb_dim
        key_padding_mask size: bsz, src_len
        attn_mask: bsz, src_len
        """

        length, bsz, _ = query.size()
        q = self.q_in_proj[energy_type].forward(query)
        q = (
            q.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )
        q = q * self.scaling
        length, bsz, _ = key.size()
        k = self.k_in_proj[energy_type].forward(key)
        k = (
            k.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        energy = torch.bmm(q, k.transpose(1, 2)) + bias

        if key_padding_mask is not None:
            energy = energy.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool), -float("inf")
            )

        return energy

    def info_energy_from_qk(
        self,
        query: Tensor,
        key: Tensor,
        energy_type: str,
        key_padding_mask: Optional[Tensor] = None,
        bias: int = 0,
    ):
        """
        Compute energy from query and key
        q_func_value is a tuple looks like
        (q_proj_func, q_tensor)
        q_tensor size: bsz, tgt_len, emb_dim
        k_tensor size: bsz, src_len, emb_dim
        key_padding_mask size: bsz, src_len
        attn_mask: bsz, src_len
        """

        tgt_len, bsz, _ = query.size()
        q = self.q_in_proj[energy_type].forward(query)
        q = q.contiguous().view(tgt_len, bsz, self.embed_dim).transpose(0, 1)
        q = q * self.scaling
        src_len, bsz, _ = key.size()
        k = self.k_in_proj[energy_type].forward(key)
        k = k.contiguous().view(src_len, bsz, self.embed_dim).transpose(0, 1)

        energy = torch.bmm(q, k.transpose(1, 2)) + bias
        energy = (
            energy.unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .contiguous()
            .view(bsz * self.num_heads, tgt_len, src_len)
        )

        if key_padding_mask is not None:
            energy = energy.masked_fill(
                key_padding_mask.unsqueeze(1).to(torch.bool), -float("inf")
            )

        return energy

    def p_choose_from_qk(
        self,
        query,
        key,
        key_padding_mask,
        train_threshold=None,
        test_threshold=None,
        incremental_state=None,
    ):
        transpose_energy = self.info_energy_from_qk(
            query,
            key,
            "transport",
            key_padding_mask=key_padding_mask,
            bias=self.energy_bias,
        )

        p_choose = torch.sigmoid(transpose_energy) + 1e-4

        return p_choose

    def p_choose(self, query, key, key_padding_mask):
        return self.p_choose_from_qk(self, query, key, key_padding_mask)

    def monotonic_attention_process_infer(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
    ):
        """
        Monotonic attention at inference time
        Notice that this function is designed for simuleval not sequence_generator
        """
        assert query is not None
        assert key is not None

        if query.size(1) != 1:
            raise RuntimeError(
                "Simultaneous translation models don't support batch decoding."
            )
        # 1. compute stepwise probability
        p_choose, p_choose_copy, p_choose_pooled, key_padding_mask_pool = self.p_choose(
            query, key, None, incremental_state
        )
        p_choose = p_choose.unsqueeze(1)

        _p_choose = p_choose[0][0].cumsum(dim=-1)

        # 2. Compute the alpha
        src_len = key.size(0)
        # Maximum steps allows in this iteration
        max_steps = src_len - 1
        monotonic_cache = self._get_monotonic_buffer(incremental_state)

        # Step for each head
        monotonic_step = monotonic_cache.get("head_step", _p_choose.new_zeros(1).long())
        assert monotonic_step is not None

        num_to_read = ((_p_choose < self.test_threshold).sum(dim=-1) + 1).clamp(
            max(
                monotonic_step.item() + 1,
                incremental_state["steps"]["tgt"] * self.pre_decision_ratio,
            ),
            max_steps + 2,
        ) + self.pre_decision_ratio

        monotonic_step = (num_to_read - 1).clamp(monotonic_step.item(), max_steps)
        monotonic_cache["head_step"] = monotonic_step
        # Whether a head is looking for new input
        monotonic_cache["head_read"] = num_to_read - 1 > max_steps
        self._set_monotonic_buffer(incremental_state, monotonic_cache)

        # 2. Update alpha
        alpha = _p_choose.new_zeros([self.num_heads, src_len]).scatter(
            1,
            (monotonic_step.unsqueeze(0))
            .repeat(self.num_heads, 1)
            .clamp(0, src_len - 1),
            1,
        )

        # 4. Compute Beta
        if self.soft_attention:
            monotonic_step = monotonic_step.t()
            beta_mask = (
                torch.arange(src_len, device=alpha.device)
                .expand_as(alpha)
                .gt(monotonic_step)
                .unsqueeze(1)
            )
            # If it's soft attention just do softmax on current context
            soft_energy = self.energy_from_qk(query, key, "soft")

            beta_float = utils.softmax(
                soft_energy.masked_fill(beta_mask, -float("inf")),
                dim=-1,
            )
            beta = beta_float.type_as(soft_energy)

            # It could happen that a head doesn't move at all
            if self.fuse_ot_with_attn:
                beta = beta * p_choose_copy
            beta = beta / beta.sum(dim=-1, keepdim=True)
        else:
            # If it's hard attention just select the last state
            beta = alpha

        return p_choose, alpha, beta

    def monotonic_attention_process_train(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        train_threshold=None,
    ):
        """
        Calculating monotonic attention process for training
        Including:
            stepwise probability: p_choose
            expected hard alignment: alpha
            expected soft attention: beta
        """
        assert query is not None
        assert key is not None

        # 1. compute stepwise probability
        p_choose, p_choose_copy, p_choose_pooled, key_padding_mask_pool = self.p_choose(
            query, key, key_padding_mask
        )

        # 2. compute expected_alignment
        alpha = rw_with_info(
            p_choose,
            pre_decision_ratio=self.pre_decision_ratio,
            train_threshold=train_threshold,
        )

        if self.mass_preservation:
            alpha = mass_preservation(alpha, key_padding_mask)

        # 3. compute expected soft attention (soft aligned model only)
        if self.soft_attention:
            soft_energy = self.energy_from_qk(
                query,
                key,
                "soft",
                key_padding_mask=None,
            )

            beta = truncated_soft_attention(
                p_choose_copy,
                alpha,
                soft_energy,
                padding_mask=key_padding_mask,
                chunk_size=self.chunk_size,
                eps=self.eps,
                fuse=self.fuse_ot_with_attn,
            )
        else:
            beta = alpha
            soft_energy = alpha

        return p_choose, alpha, beta, soft_energy, key_padding_mask_pool

    def monotonic_attention_process_test(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
    ):
        """
        Calculating monotonic attention process for training
        Including:
            stepwise probability: p_choose
            expected hard alignment: alpha
            expected soft attention: beta
        """
        assert query is not None
        assert key is not None

        # 1. compute stepwise probability
        p_choose, p_choose_copy, p_choose_pooled, key_padding_mask_pool = self.p_choose(
            query, key, key_padding_mask
        )

        # 2. compute expected_alignment
        alpha = rw_with_info(
            p_choose,
            pre_decision_ratio=self.pre_decision_ratio,
            test_threshold=self.test_threshold,
        )

        # 3. compute expected soft attention (soft aligned model only)
        if self.soft_attention:
            soft_energy = self.energy_from_qk(
                query,
                key,
                "soft",
                key_padding_mask=None,
            )

            beta = truncated_soft_attention(
                p_choose_copy,
                alpha,
                soft_energy,
                padding_mask=key_padding_mask,
                chunk_size=self.chunk_size,
                eps=self.eps,
                fuse=self.fuse_ot_with_attn,
            )
        else:
            beta = alpha
            soft_energy = alpha

        return p_choose, alpha, beta, soft_energy, key_padding_mask_pool

    def forward(
        self,
        query: Optional[Tensor],
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        need_head_weights: bool = False,
        train_threshold=None,
        step=None,
        global_wait_k=None,
    ):
        """
        query: tgt_len, bsz, embed_dim
        key: src_len, bsz, embed_dim
        value: src_len, bsz, embed_dim
        """

        assert attn_mask is None
        assert query is not None
        assert key is not None
        assert value is not None

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)

        if key_padding_mask is not None:
            assert not key_padding_mask[:, 0].any(), "Only right padding is supported."
            key_padding_mask = (
                key_padding_mask.unsqueeze(1)
                .expand([bsz, self.num_heads, src_len])
                .contiguous()
                .view(-1, src_len)
            )

        if step is not None:
            (
                p_choose,
                alpha,
                beta,
                soft_energy,
                key_padding_mask_pool,
            ) = self.monotonic_attention_process_test(query, key, key_padding_mask)
        elif incremental_state is not None:
            # Inference
            (p_choose, alpha, beta) = self.monotonic_attention_process_infer(
                query, key, incremental_state
            )
            soft_energy = beta
            key_padding_mask_pool = None
        else:
            # Train
            (
                p_choose,
                alpha,
                beta,
                soft_energy,
                key_padding_mask_pool,
            ) = self.monotonic_attention_process_train(
                query, key, key_padding_mask, train_threshold
            )

        v = self.v_proj(value)

        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn = torch.bmm(beta.type_as(v), v)

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, -1)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, {
            "p_choose": p_choose,
            "alpha": alpha,
            "beta": beta,
            "key_padding_mask_pool": key_padding_mask_pool,
        }

    def _get_monotonic_buffer(
        self, incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    ):
        maybe_incremental_state = self.get_incremental_state(
            incremental_state,
            "monotonic",
        )
        if maybe_incremental_state is None:
            typed_empty_dict: Dict[str, Optional[Tensor]] = {}
            return typed_empty_dict
        else:
            return maybe_incremental_state

    def _set_monotonic_buffer(
        self,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        buffer: Dict[str, Optional[Tensor]],
    ):
        self.set_incremental_state(
            incremental_state,
            "monotonic",
            buffer,
        )
