# Speech-to-text simultaneous translation based on MoE wait-k policy
# Refer https://aclanthology.org/2021.emnlp-main.581/ for details
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

from .monotonic_multihead_attention import (
    MonotonicAttention,
    MonotonicInfiniteLookbackAttention,
    WaitKAttention,
)

from fairseq.modules import MultiheadAttention

from . import register_monotonic_attention
from typing import Dict, Optional


@register_monotonic_attention("moe_waitk")
class MoEWaitKAttention(WaitKAttention):
    """
    STACL: Simultaneous Translation with Implicit Anticipation and
    Controllable Latency using Prefix-to-Prefix Framework
    https://www.aclweb.org/anthology/P19-1289/
    """

    def __init__(self, args):
        super().__init__(args)
        # self.gate_attn_proj = nn.Linear(self.num_heads,self.num_heads, bias=True)
        self.gate_proj = nn.Linear(self.num_heads + 1, self.num_heads, bias=True)

        if self.num_heads == 16:
            self.head_k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        if self.num_heads == 8:
            self.head_k = [1, 3, 5, 7, 9, 11, 13, 15]
        if self.num_heads == 4:
            self.head_k = [1, 6, 11, 16]
        self.avg_gate = getattr(args, "avg_gate", False)

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
        bsz = query.size(1)
        tgt_len = query.size(0)
        src_len = key.size(0)
        tmp = (
            torch.arange(1, src_len + 1, device=query.device)
            .unsqueeze(0)
            .type_as(query)
        )
        p_chooses = []
        for i in range(0, self.num_heads):

            index = (
                min(self.waitk_lagging, self.head_k[i])
                + incremental_state["steps"]["tgt"]
                - 1
            ) * self.pre_decision_ratio

            p_choose = (
                (tmp == index)
                .type_as(query)
                .unsqueeze(0)
                .repeat(bsz, 1, 1)
                .unsqueeze(1)
            )
            p_chooses.append(p_choose)
        p_choose = (
            torch.cat(p_chooses, dim=1)
            .contiguous()
            .view(bsz * self.num_heads, tgt_len, src_len)
        )

        alpha = p_choose.contiguous()

        index = (
            self.waitk_lagging + incremental_state["steps"]["tgt"] - 1
        ) * self.pre_decision_ratio
        p_choose = (
            (tmp == index)
            .type_as(query)
            .unsqueeze(0)
            .repeat(bsz * self.num_heads, 1, 1)
        )
        alpha_gate = p_choose.contiguous()

        p_choose = p_choose.squeeze(1)

        # 2. Compute the alpha
        src_len = key.size(0)
        # Maximum steps allows in this iteration
        max_steps = src_len - 1 if self.mass_preservation else src_len
        monotonic_cache = self._get_monotonic_buffer(incremental_state)
        # Step for each head
        monotonic_step = monotonic_cache.get(
            "head_step", p_choose.new_zeros(1, self.num_heads).long()
        )
        assert monotonic_step is not None
        finish_read = monotonic_step.eq(max_steps)
        p_choose_i = torch.tensor(1)

        while finish_read.sum().item() < self.num_heads:
            # p_choose: self.num_heads, src_len
            # only choose the p at monotonic steps
            # p_choose_i: 1, self.num_heads
            p_choose_i = p_choose.gather(
                1,
                monotonic_step.clamp(0, src_len - 1).t(),
            ).t()

            read_one_step = (
                (p_choose_i < 0.5).type_as(monotonic_step).masked_fill(finish_read, 0)
            )
            # 1 x bsz
            # sample actions on unfinished seq
            # 0 means stay, finish reading
            # 1 means leave, continue reading

            monotonic_step += read_one_step
            finish_read = monotonic_step.eq(max_steps) | (read_one_step == 0)

        # p_choose at last steps
        p_choose_i = p_choose.gather(
            1,
            monotonic_step.clamp(0, src_len - 1).t(),
        ).t()

        monotonic_cache["head_step"] = monotonic_step
        # Whether a head is looking for new input
        monotonic_cache["head_read"] = monotonic_step.eq(max_steps) & (p_choose_i < 0.5)
        self._set_monotonic_buffer(incremental_state, monotonic_cache)

        if not self.mass_preservation:
            alpha = alpha.masked_fill(
                (monotonic_step == max_steps).view(self.num_heads, 1), 0
            )

        moe_monotonic_step = ((alpha.cummax(dim=-1)[0] == 0).sum(dim=-1) + 1).clamp(
            1, src_len
        ) - 1
        beta_mask = (
            torch.arange(src_len, device=alpha.device)
            .expand_as(alpha)
            .gt(moe_monotonic_step.unsqueeze(1))
        )
        # If it's soft attention just do softmax on current context
        soft_energy = self.energy_from_qk(query, key, "soft")
        beta = torch.nn.functional.softmax(
            soft_energy.masked_fill(beta_mask, -float("inf")), dim=-1
        )

        monotonic_step = monotonic_step.t()
        beta_gate_mask = (
            torch.arange(src_len, device=alpha.device)
            .expand_as(alpha)
            .gt(monotonic_step.unsqueeze(1))
        )

        beta_gate = torch.nn.functional.softmax(
            soft_energy.masked_fill(beta_gate_mask, -float("inf")), dim=-1
        )

        return p_choose, alpha, beta, beta_gate, 1 - beta_mask.type_as(beta)

    def monotonic_attention_process_train(
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

        bsz = query.size(1)
        tgt_len = query.size(0)
        src_len = key.size(0)
        tmp = (
            torch.arange(1, src_len + 1, device=query.device)
            .unsqueeze(0)
            .type_as(query)
        )
        p_chooses = []
        for i in range(0, self.num_heads):
            index = (
                (
                    torch.arange(
                        min(self.waitk_lagging, self.head_k[i]),
                        tgt_len + min(self.waitk_lagging, self.head_k[i]),
                        device=query.device,
                    )
                    * self.pre_decision_ratio
                )
                .clamp(0, src_len)
                .unsqueeze(1)
                .type_as(query)
            )
            p_choose = (
                (tmp == index)
                .type_as(query)
                .unsqueeze(0)
                .repeat(bsz, 1, 1)
                .unsqueeze(1)
            )
            p_chooses.append(p_choose)
        p_choose = (
            torch.cat(p_chooses, dim=1)
            .contiguous()
            .view(bsz * self.num_heads, tgt_len, src_len)
        )

        alpha = p_choose.contiguous()

        index = (
            (
                torch.arange(
                    self.waitk_lagging,
                    tgt_len + self.waitk_lagging,
                    device=query.device,
                )
                * self.pre_decision_ratio
            )
            .clamp(0, src_len)
            .unsqueeze(1)
            .type_as(query)
        )
        p_choose_gate = (
            (tmp == index)
            .type_as(query)
            .unsqueeze(0)
            .repeat(bsz * self.num_heads, 1, 1)
        )
        alpha_gate = p_choose_gate.contiguous()

        if self.mass_preservation:
            alpha = mass_preservation(alpha, key_padding_mask)
            alpha_gate = mass_preservation(alpha_gate, key_padding_mask)

        # 3. compute expected soft attention (soft aligned model only)

        soft_energy = self.energy_from_qk(
            query,
            key,
            "soft",
            key_padding_mask=None,
        )

        beta = expected_soft_attention(
            alpha,
            soft_energy,
            padding_mask=key_padding_mask,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )

        beta_gate = expected_soft_attention(
            alpha_gate,
            soft_energy,
            padding_mask=key_padding_mask,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )

        moe_monotonic_step = ((alpha.cummax(dim=-1)[0] == 0).sum(dim=-1) + 1).clamp(
            1, src_len
        ) - 1
        beta_mask = (
            torch.arange(src_len, device=alpha.device)
            .expand_as(alpha)
            .gt(moe_monotonic_step.unsqueeze(-1))
        )

        return (
            p_choose,
            alpha,
            beta,
            beta_gate,
            1 - beta_mask.type_as(beta),
            soft_energy,
        )

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

        bsz = query.size(1)
        tgt_len = query.size(0)
        src_len = key.size(0)
        tmp = (
            torch.arange(1, src_len + 1, device=query.device)
            .unsqueeze(0)
            .type_as(query)
        )
        p_chooses = []
        for i in range(0, self.num_heads):

            index = (
                (
                    torch.arange(
                        min(self.waitk_lagging, self.head_k[i]),
                        tgt_len + min(self.waitk_lagging, self.head_k[i]),
                        device=query.device,
                    )
                    * self.pre_decision_ratio
                )
                .clamp(0, src_len)
                .unsqueeze(1)
                .type_as(query)
            )
            p_choose = (
                (tmp == index)
                .type_as(query)
                .unsqueeze(0)
                .repeat(bsz, 1, 1)
                .unsqueeze(1)
            )
            p_chooses.append(p_choose)
        p_choose = (
            torch.cat(p_chooses, dim=1)
            .contiguous()
            .view(bsz * self.num_heads, tgt_len, src_len)
        )

        alpha = p_choose.contiguous()

        index = (
            (
                torch.arange(
                    self.waitk_lagging,
                    tgt_len + self.waitk_lagging,
                    device=query.device,
                )
                * self.pre_decision_ratio
            )
            .clamp(0, src_len)
            .unsqueeze(1)
            .type_as(query)
        )
        p_choose_gate = (
            (tmp == index)
            .type_as(query)
            .unsqueeze(0)
            .repeat(bsz * self.num_heads, 1, 1)
        )
        alpha_gate = p_choose_gate.contiguous()

        if self.mass_preservation:
            alpha = mass_preservation(alpha, key_padding_mask)
            alpha_gate = mass_preservation(alpha_gate, key_padding_mask)

        # 3. compute expected soft attention (soft aligned model only)

        soft_energy = self.energy_from_qk(
            query,
            key,
            "soft",
            key_padding_mask=None,
        )

        beta = expected_soft_attention(
            alpha,
            soft_energy,
            padding_mask=key_padding_mask,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )

        beta_gate = expected_soft_attention(
            alpha_gate,
            soft_energy,
            padding_mask=key_padding_mask,
            chunk_size=self.chunk_size,
            eps=self.eps,
        )

        moe_monotonic_step = ((alpha.cummax(dim=-1)[0] == 0).sum(dim=-1) + 1).clamp(
            1, src_len
        ) - 1
        beta_mask = (
            torch.arange(src_len, device=alpha.device)
            .expand_as(alpha)
            .gt(moe_monotonic_step.unsqueeze(-1))
        )

        return (
            p_choose_gate,
            alpha,
            beta,
            beta_gate,
            1 - beta_mask.type_as(beta),
            soft_energy,
        )

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

        gate_k = key

        tgt_len, bsz, embed_dim = query.size()
        src_len = value.size(0)
        if global_wait_k is not None:
            self.waitk_lagging = global_wait_k

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
                beta_gate,
                expert_mask,
                soft_energy,
            ) = self.monotonic_attention_process_test(query, key, incremental_state)

        elif incremental_state is not None:
            # Inference
            (
                p_choose,
                alpha,
                beta,
                beta_gate,
                expert_mask,
            ) = self.monotonic_attention_process_infer(query, key, incremental_state)
            soft_energy = beta
        else:
            # Train
            (
                p_choose,
                alpha,
                beta,
                beta_gate,
                expert_mask,
                soft_energy,
            ) = self.monotonic_attention_process_train(query, key, key_padding_mask)

        v = self.v_proj(value)
        length, bsz, _ = v.size()
        v = (
            v.contiguous()
            .view(length, bsz * self.num_heads, self.head_dim)
            .transpose(0, 1)
        )

        attn = torch.bmm(beta.type_as(v), v)

        if self.avg_gate:
            gate = torch.ones_like(attn)
        else:
            gate = self.predict_gate(beta_gate, expert_mask)

        attn = torch.mul(attn, gate.type_as(attn))

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        attn = self.out_proj(attn)

        p_choose = p_choose.view(bsz, self.num_heads, tgt_len, src_len)
        alpha = alpha.view(bsz, self.num_heads, tgt_len, src_len)
        beta = beta.view(bsz, self.num_heads, tgt_len, src_len)

        return attn, {
            "p_choose": p_choose,
            "alpha": alpha,
            "beta": beta,
        }

    def predict_gate(self, gate_attn_weights, expert_mask):
        bsz, tgt_len, src_len = gate_attn_weights.size()
        bsz = bsz // self.num_heads
        expert_mask = expert_mask.contiguous().view(
            bsz, self.num_heads, tgt_len, src_len
        )

        gate_attn_weights = gate_attn_weights.contiguous().view(
            bsz, self.num_heads, tgt_len, src_len
        )
        gate_attn_weights = gate_attn_weights * expert_mask
        gate_attn_weights = (
            gate_attn_weights.sum(dim=3).contiguous().view(bsz, self.num_heads, tgt_len)
        )
        gate_attn_weights = gate_attn_weights.contiguous() / expert_mask.sum(
            dim=3
        ).contiguous().view(-1, self.num_heads, tgt_len)
        gate_attn_weights = gate_attn_weights.contiguous().view(
            bsz * self.num_heads, tgt_len
        )
        gate_attn_weights = (
            gate_attn_weights.contiguous()
            .view(bsz, self.num_heads, tgt_len)
            .transpose(1, 2)
            .contiguous()
            .view(bsz * tgt_len, self.num_heads)
            .type_as(gate_attn_weights)
        )

        gate_x = torch.cat(
            (
                gate_attn_weights,
                torch.full(
                    (bsz * tgt_len, 1),
                    self.waitk_lagging,
                    device=gate_attn_weights.device,
                ),
            ),
            dim=1,
        )

        gate = torch.tanh(self.gate_proj(gate_x))
        gate = gate.contiguous().view(bsz, tgt_len, self.num_heads).transpose(1, 2)
        gate_float = utils.softmax(gate, dim=1, onnx_trace=self.onnx_trace)
        gate = gate_float.type_as(gate)
        gate = (
            gate.unsqueeze(3)
            .repeat(1, 1, 1, self.head_dim)
            .contiguous()
            .view(bsz * self.num_heads, tgt_len, self.head_dim)
        )
        gate = gate * self.num_heads

        return gate
