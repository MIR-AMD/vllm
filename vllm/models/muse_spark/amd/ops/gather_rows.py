# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Row gather whose cost follows a device-side row count, not the buffer size."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _gather_rows_kernel(
    src_ptr,
    idx_ptr,
    count_ptr,
    out_ptr,
    row_elems,
    BLOCK: tl.constexpr,
):
    j = tl.program_id(0)
    if j >= tl.load(count_ptr):
        return
    src = src_ptr + tl.load(idx_ptr + j).to(tl.int64) * row_elems
    dst = out_ptr + j.to(tl.int64) * row_elems
    for start in range(0, row_elems, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < row_elems
        tl.store(dst + offs, tl.load(src + offs, mask=mask), mask=mask)


def gather_rows(
    src: torch.Tensor, index: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    """``out[j] = src[index[j]]`` for ``j < count[0]``; later rows are undefined.

    Rows are copied as raw bytes, so packed dtypes (e.g. MXFP4) work.
    """
    assert src.dim() == 2 and src.is_contiguous()
    raw = src.view(torch.uint8)
    if raw.shape[1] % 4 == 0:
        raw = raw.view(torch.int32)
    out = raw.new_empty((index.shape[0], raw.shape[1]))
    if not src.is_cuda:
        out.copy_(raw.index_select(0, index.long()))
    elif index.shape[0] > 0:
        _gather_rows_kernel[(index.shape[0],)](
            raw, index, count, out, raw.shape[1], BLOCK=1024
        )
    return out.view(torch.uint8).view(src.dtype)
