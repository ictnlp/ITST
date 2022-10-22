# truncated attention: only focus on a perfix of source sequence
#
# by Shaolei Zhang


from typing import Optional
import torch
from torch import Tensor
from fairseq import utils

from examples.simultaneous_translation.utils.functions import (
    exclusive_cumprod,
    prob_check,
    moving_sum,
)


def rw_with_info(
    p_choose: Tensor,
    pre_decision_ratio=7,
    padding_mask: Optional[Tensor] = None,
    eps: float = 1e-6,
    train_threshold=None,
    test_threshold=None,
):
    """
    Expected input size
    p_choose: bsz, tgt_len, src_len
    """
    # p_choose: bsz, tgt_len, src_len
    bsz, tgt_len, src_len = p_choose.size()

    if train_threshold is not None:
        max_id = (p_choose.cumsum(dim=-1) < train_threshold).sum(
            dim=-1, keepdim=True
        ) + 1
        max_id = torch.cummax(max_id, dim=-2)[0]
        max_id = max_id.max(
            (torch.arange(1, tgt_len + 1, device=p_choose.device) * pre_decision_ratio)
            .unsqueeze(0)
            .unsqueeze(2)
        ).clamp(1, src_len)
        max_id = (max_id + pre_decision_ratio).clamp(1, src_len)
        info_mask = (
            torch.arange(1, src_len + 1, device=max_id.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(bsz, tgt_len, 1)
        )
        alpha = (info_mask == max_id).type_as(p_choose)

    if test_threshold is not None:
        max_id = (p_choose.cumsum(dim=-1) < test_threshold).sum(
            dim=-1, keepdim=True
        ) + 1
        max_id = torch.cummax(max_id, dim=-2)[0]
        max_id = max_id.max(
            (torch.arange(1, tgt_len + 1, device=p_choose.device) * pre_decision_ratio)
            .unsqueeze(0)
            .unsqueeze(2)
        )
        max_id = (max_id + pre_decision_ratio).clamp(1, src_len)
        info_mask = (
            torch.arange(1, src_len + 1, device=max_id.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(bsz, tgt_len, 1)
        )
        alpha = (info_mask == max_id).type_as(p_choose)
    if test_threshold is None and train_threshold is None:
        alpha = torch.zeros_like(p_choose)
        alpha[:, :, -1] = 1

    return alpha


def truncated_soft_attention(
    p_choose_copy,
    alpha: Tensor,
    soft_energy: Tensor,
    padding_mask: Optional[Tensor] = None,
    chunk_size: Optional[int] = None,
    eps: float = 1e-10,
    fuse=True,
):
    """
    alpha: bsz, tgt_len, src_len
    soft_energy: bsz, tgt_len, src_len
    padding_mask: bsz, src_len
    left_padding: bool
    """

    bsz, tgt_len, src_len = alpha.size()
    truncated_mask = torch.cummax(alpha.flip(dims=[2]), dim=2)[0].flip(dims=[2])
    soft_energy = soft_energy.masked_fill((1 - truncated_mask).bool(), -float("inf"))

    if padding_mask is not None:
        soft_energy = soft_energy.masked_fill(padding_mask.unsqueeze(1), -float("inf"))

    prob_check(alpha)

    beta_float = utils.softmax(
        soft_energy,
        dim=-1,
    )
    beta = beta_float.type_as(soft_energy)

    if fuse:
        beta = beta * p_choose_copy
    beta = beta / beta.sum(dim=-1, keepdim=True)

    if padding_mask is not None:
        beta = beta.masked_fill(padding_mask.unsqueeze(1).to(torch.bool), 0.0)

    # Mix precision to prevent overflow for fp16
    # beta = beta.type(dtype)

    prob_check(beta)

    return beta
