# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm MXFP4 under DP + EP with the MoRI EPv1 all2all.

MoRI sends a token once to every rank owning one of its experts and its combine
only sums the rows those ranks return, so each expert rank must finalize its
own routes: ``sum_k w_k * norm(out_k)`` over its local experts.
"""

from collections.abc import Callable
from dataclasses import fields

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXMoEQuantConfig,
    MuseSparkTopKWeightAndReduce,
    finalizes_before_mori_combine,
    validate_bf16_deployment,
    validate_fp8_deployment,
    validate_mxfp4_deployment,
)


@pytest.fixture(autouse=True)
def _aiter_fused_moe_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # FusedMoEConfig asserts MoRI EPv1 runs with AITER fused MoE.
    monkeypatch.setattr(rocm_aiter_ops, "is_fused_moe_enabled", lambda: True)


def _moe_config(
    *,
    tp_size: int = 1,
    dp_size: int = 4,
    pcp_size: int = 1,
    sp_size: int = 1,
    all2all_backend: str = "mori_high_throughput",
    mori_ep_v2: bool = False,
    use_ep: bool = True,
    enable_eplb: bool = False,
    num_experts: int = 16,
    top_k: int = 4,
    hidden_size: int = 64,
) -> FusedMoEConfig:
    # As FusedMoEParallelConfig.make: under EP the attention TP folds into EP.
    ep_size = tp_size * dp_size * pcp_size if use_ep else 1
    parallel_config = FusedMoEParallelConfig(
        tp_size=1 if use_ep else tp_size * dp_size * pcp_size,
        pcp_size=pcp_size,
        dp_size=dp_size,
        ep_size=ep_size,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=sp_size,
        use_ep=use_ep,
        all2all_backend=all2all_backend,
        enable_eplb=enable_eplb,
        mori_ep_v2=mori_ep_v2,
    )
    return FusedMoEConfig(
        num_experts=num_experts,
        experts_per_token=top_k,
        hidden_dim=hidden_size,
        intermediate_size=2 * hidden_size,
        num_local_experts=num_experts // ep_size,
        num_logical_experts=num_experts,
        activation=MoEActivation.SILU,
        device=torch.device("cpu"),
        routing_method=RoutingMethodType.Custom,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
        moe_backend="triton",
    )


# "mori" runs EPv1 unless VLLM_MORI_EP_V2 is set.
@pytest.mark.parametrize(
    "backend", ["mori_high_throughput", "mori_low_latency", "mori"]
)
@pytest.mark.parametrize("dp_size", [8, 16, 32])
def test_mxfp4_accepts_tp1_dp_ep_mori(backend: str, dp_size: int) -> None:
    config = _moe_config(dp_size=dp_size, all2all_backend=backend, num_experts=128)
    assert finalizes_before_mori_combine(config)
    validate_mxfp4_deployment(config)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        # The naive all2all finalizes after gathering every route; not wired up.
        ({"all2all_backend": "allgather_reducescatter"}, "DP1"),
        # EPv2 is a separate prepare/finalize, not covered here.
        ({"all2all_backend": "mori", "mori_ep_v2": True}, "DP1"),
        # Attention TP > 1 (two EP ranks per DP rank) is not validated.
        ({"tp_size": 2, "dp_size": 2}, "DP1"),
        # DP without EP has no all2all at all.
        ({"use_ep": False}, "DP1"),
        ({"pcp_size": 2, "dp_size": 1}, "DP1"),
        ({"sp_size": 2}, "sequence"),
        ({"enable_eplb": True}, "EPLB"),
    ],
)
def test_mxfp4_rejects_other_dp_deployments(kwargs: dict, reason: str) -> None:
    config = _moe_config(**kwargs)
    with pytest.raises(NotImplementedError, match=reason):
        validate_mxfp4_deployment(config)


@pytest.mark.parametrize(
    "validate", [validate_bf16_deployment, validate_fp8_deployment]
)
def test_bf16_and_fp8_still_require_dp1(
    validate: Callable[[FusedMoEConfig], None],
) -> None:
    with pytest.raises(NotImplementedError, match="DP1"):
        validate(_moe_config())


def _rms_norm_with_input_scale(
    gain: torch.Tensor, eps: float
) -> Callable[..., torch.Tensor]:
    """``rms_norm(x * gain)`` like MuseSparkRMSNormWithInputScale, row-wise."""

    def norm(x: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        y = x.float() * gain
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        return y.to(out_dtype or x.dtype)

    return norm


def _experts(config: FusedMoEConfig, norm_fn) -> MuseSparkAiterExperts:
    quant_config = MuseSparkQuarkOCPMXMoEQuantConfig(
        **{
            f.name: getattr(FUSED_MOE_UNQUANTIZED_CONFIG, f.name)
            for f in fields(FusedMoEQuantConfig)
            if f.init
        },
        post_expert_norm_fn=norm_fn,
    )
    return MuseSparkAiterExperts(config, quant_config)


def test_dp1_experts_keep_the_route_major_finalize() -> None:
    config = _moe_config(dp_size=1, use_ep=False)
    experts = _experts(config, _rms_norm_with_input_scale(torch.ones(64), 1e-6))
    assert experts._local_finalize is None
    shapes = experts.workspace_shapes(
        5, 128, 64, 4, 16, 16, None, MoEActivation.SILU
    )
    assert shapes[2] == (5, 4, 64)
    assert isinstance(
        experts.finalize_weight_and_reduce_impl(), MuseSparkTopKWeightAndReduce
    )


def _simulate_mori_ep(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ep_size: int,
    num_experts: int,
    top_k: int,
    hidden: int,
    tokens_per_rank: int,
    seed: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run MuseSparkAiterExperts.apply on every simulated EP rank behind a
    MoRI-style dispatch/combine. Returns (combined, reference) per source rank."""
    gen = torch.Generator().manual_seed(seed)
    local = num_experts // ep_size
    eps = 1e-6
    gain = torch.rand(hidden, generator=gen) + 0.5
    norm_fn = _rms_norm_with_input_scale(gain, eps)
    expert_w = torch.randn(num_experts, hidden, hidden, generator=gen) / hidden**0.5

    x = [torch.randn(tokens_per_rank, hidden, generator=gen) for _ in range(ep_size)]
    ids = [
        torch.stack(
            [
                torch.randperm(num_experts, generator=gen)[:top_k]
                for _ in range(tokens_per_rank)
            ]
        ).to(torch.int32)
        for _ in range(ep_size)
    ]
    # Token 0 of rank 0 routes only to rank 0's experts: other ranks get nothing.
    assert top_k <= local
    ids[0][0] = torch.arange(top_k, dtype=torch.int32)
    w = [
        torch.softmax(torch.randn(tokens_per_rank, top_k, generator=gen), -1)
        for _ in range(ep_size)
    ]

    reference = [
        sum(
            w[s][:, k : k + 1]
            * norm_fn(
                torch.einsum("th,thg->tg", x[s], expert_w[ids[s][:, k].long()]),
                out_dtype=torch.float32,
            )
            for k in range(top_k)
        )
        for s in range(ep_size)
    ]

    capacity = ep_size * tokens_per_rank
    combined = [torch.zeros(tokens_per_rank, hidden) for _ in range(ep_size)]
    calls: list[dict] = []

    def _fake_aiter(**kwargs) -> torch.Tensor:
        hs = kwargs["hidden_states"]
        flat_ids = kwargs["topk_ids"]
        mask = kwargs["expert_mask"]
        n = int(kwargs["num_local_tokens"])
        assert flat_ids.shape == (hs.shape[0], 2)
        assert bool((flat_ids[:, 1] == mask.shape[0] - 1).all())
        assert bool((kwargs["topk_weights"] == 1).all())
        assert kwargs["apply_router_weight_on_input"] is False
        calls.append(kwargs)
        # Routes AITER skips (remote experts, rows past the receive count)
        # are left unwritten: poison them.
        out = torch.full((hs.shape[0], hidden), float("nan"))
        for i in range(n):
            e = int(flat_ids[i, 0])
            if mask[e]:
                out[i] = hs[i].float() @ expert_w[e]
        return out.to(kwargs["output_dtype"])

    monkeypatch.setattr(
        "vllm.models.muse_spark.amd.muse_moe_runner.rocm_aiter_fused_experts",
        _fake_aiter,
    )

    config = _moe_config(
        dp_size=ep_size, num_experts=num_experts, top_k=top_k, hidden_size=hidden
    )
    for r in range(ep_size):
        owned = range(r * local, (r + 1) * local)
        # MoRI dispatch: each source token once per destination rank.
        received = [
            (s, t)
            for s in range(ep_size)
            for t in range(tokens_per_rank)
            if any(int(e) in owned for e in ids[s][t])
        ]
        n = len(received)
        # Capacity-sized receive buffers whose tail holds stale garbage.
        recv_x = torch.randn(capacity, hidden, generator=gen)
        recv_ids = torch.randint(
            10**6, 2 * 10**6, (capacity, top_k), generator=gen, dtype=torch.int32
        )
        recv_w = torch.rand(capacity, top_k, generator=gen)
        recv_scale = torch.randint(
            0, 256, (capacity, hidden // 32), generator=gen, dtype=torch.uint8
        )
        for j, (s, t) in enumerate(received):
            recv_x[j], recv_ids[j], recv_w[j] = x[s][t], ids[s][t], w[s][t]
        expert_mask = torch.zeros(num_experts + 1, dtype=torch.int32)
        expert_mask[list(owned)] = 1

        experts = _experts(config, norm_fn)
        experts._extra_fused_moe_kwargs = lambda: {}  # type: ignore[method-assign]
        assert experts._local_finalize is not None
        assert isinstance(
            experts.finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP
        )
        out_shape = experts.workspace_shapes(
            capacity,
            2 * hidden,
            hidden,
            top_k,
            num_experts,
            local,
            None,
            MoEActivation.SILU,
        )[2]
        assert out_shape == (capacity, hidden)
        output = torch.full(out_shape, float("nan"))
        experts.apply(
            output,
            recv_x,
            torch.empty(0),
            torch.empty(0),
            recv_w,
            recv_ids,
            MoEActivation.SILU,
            num_experts,
            expert_mask,
            recv_scale,
            None,
            torch.empty(0),
            torch.empty(0),
            mk.ExpertTokensMetadata(
                expert_num_tokens=torch.tensor([n], dtype=torch.int32),
                expert_num_tokens_cpu=None,
            ),
            False,
        )
        call = calls[-1]
        torch.testing.assert_close(
            call["hidden_states"], recv_x.repeat_interleave(top_k, dim=0)
        )
        torch.testing.assert_close(
            call["a1q_scale"], recv_scale.repeat_interleave(top_k, dim=0)
        )
        assert int(call["num_local_tokens"]) == n * top_k
        assert experts._take_route_output() is None
        # Every row, including the stale tail, is finite.
        assert bool(torch.isfinite(output).all())
        # MoRI combine: an unweighted sum of the returned rows per source token.
        for j, (s, t) in enumerate(received):
            combined[s][t] += output[j]
    return combined, reference


@pytest.mark.parametrize(
    "ep_size,num_experts,top_k", [(2, 8, 2), (4, 16, 4), (8, 64, 6)]
)
def test_ep_rank_partials_sum_to_the_dp1_output(
    monkeypatch: pytest.MonkeyPatch, ep_size: int, num_experts: int, top_k: int
) -> None:
    combined, reference = _simulate_mori_ep(
        monkeypatch,
        ep_size=ep_size,
        num_experts=num_experts,
        top_k=top_k,
        hidden=64,
        tokens_per_rank=5,
        seed=ep_size,
    )
    for got, ref in zip(combined, reference):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)
