# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm MXFP4 under DP + EP with the MoRI EPv1 all2all.

MoRI sends a token once to every rank owning one of its experts and its combine
only sums the rows those ranks return, so each expert rank must finalize its
own routes: ``sum_k w_k * norm(out_k)`` over its local experts.
"""

from collections.abc import Callable
from dataclasses import fields

from types import SimpleNamespace

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
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import Mxfp4MoeBackend
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.quark.quark_moe import (
    QuarkOCP_MX_MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kMxfp4Dynamic,
    kMxfp4Static,
)
from vllm.models.muse_spark.amd.model import MuseSparkMoE
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXFusedMoE,
    MuseSparkQuarkOCPMXMoEQuantConfig,
    MuseSparkTopKWeightAndReduce,
    MuseSparkTritonExperts,
    MuseSparkUnquantizedFusedMoE,
    MuseSparkUnquantizedMoEQuantConfig,
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
    moe_backend: str = "triton",
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
        moe_backend=moe_backend,
    )


# "mori" runs EPv1 unless VLLM_MORI_EP_V2 is set.
@pytest.mark.parametrize(
    "backend", ["mori_high_throughput", "mori_low_latency", "mori"]
)
@pytest.mark.parametrize("dp_size", [8, 16, 32])
def test_mxfp4_and_bf16_accept_tp1_dp_ep_mori(backend: str, dp_size: int) -> None:
    config = _moe_config(dp_size=dp_size, all2all_backend=backend, num_experts=128)
    assert finalizes_before_mori_combine(config)
    validate_mxfp4_deployment(config)
    # Muse Spark's first and last MoE layers stay BF16 in the MXFP4 checkpoint.
    validate_bf16_deployment(config)


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
@pytest.mark.parametrize(
    "validate", [validate_mxfp4_deployment, validate_bf16_deployment]
)
def test_other_dp_deployments_are_rejected(
    validate: Callable[[FusedMoEConfig], None], kwargs: dict, reason: str
) -> None:
    config = _moe_config(**kwargs)
    with pytest.raises(NotImplementedError, match=reason):
        validate(config)


def test_fp8_still_requires_dp1() -> None:
    with pytest.raises(NotImplementedError, match="DP1"):
        validate_fp8_deployment(_moe_config())


@pytest.mark.parametrize("moe_backend", ["aiter", "flashinfer_trtllm"])
def test_bf16_under_mori_still_requires_the_triton_backend(moe_backend: str) -> None:
    with pytest.raises(NotImplementedError, match="Triton"):
        validate_bf16_deployment(_moe_config(moe_backend=moe_backend))


def _rms_norm_with_input_scale(
    gain: torch.Tensor, eps: float
) -> Callable[..., torch.Tensor]:
    """``rms_norm(x * gain)`` like MuseSparkRMSNormWithInputScale, row-wise."""

    def norm(x: torch.Tensor, *, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        y = x.float() * gain
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
        return y.to(out_dtype or x.dtype)

    return norm


def _experts(
    config: FusedMoEConfig, norm_fn, backend: str = "mxfp4"
) -> MuseSparkAiterExperts | MuseSparkTritonExperts:
    config_cls, experts_cls = {
        "mxfp4": (MuseSparkQuarkOCPMXMoEQuantConfig, MuseSparkAiterExperts),
        "bf16": (MuseSparkUnquantizedMoEQuantConfig, MuseSparkTritonExperts),
    }[backend]
    quant_config = config_cls(
        **{
            f.name: getattr(FUSED_MOE_UNQUANTIZED_CONFIG, f.name)
            for f in fields(FusedMoEQuantConfig)
            if f.init
        },
        post_expert_norm_fn=norm_fn,
    )
    return experts_cls(config, quant_config)


@pytest.mark.parametrize("backend", ["mxfp4", "bf16"])
def test_dp1_experts_keep_the_route_major_finalize(backend: str) -> None:
    config = _moe_config(dp_size=1, use_ep=False)
    experts = _experts(
        config, _rms_norm_with_input_scale(torch.ones(64), 1e-6), backend
    )
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
    backend: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the Muse experts' apply on every simulated EP rank behind a
    MoRI-style dispatch/combine. Returns (combined, reference) per source rank.

    ``mxfp4`` runs MuseSparkAiterExperts with AITER faked; ``bf16`` runs
    MuseSparkTritonExperts with the stock Triton GEMMs faked."""
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

    def _fake_triton_apply(
        self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
        global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
        workspace2, expert_tokens_meta, apply_router_weight_on_input,
    ) -> None:  # fmt: skip
        assert not self._apply_router_weight_in_gemm2(apply_router_weight_on_input)
        assert output.shape == hidden_states.shape
        calls.append({"topk_ids": topk_ids.clone(), "a1q_scale": a1q_scale})
        # GEMM2 rows of skipped routes (remote, padding) are never written.
        rows = torch.full((*topk_ids.shape, hidden), float("nan"))
        for i, k in (topk_ids >= 0).nonzero().tolist():
            e = int(topk_ids[i, k])
            if expert_map[e] >= 0:
                rows[i, k] = hidden_states[i].float() @ expert_w[e]
        self._finalize_expert_output(rows, output, topk_ids, expert_map)

    monkeypatch.setattr(TritonExperts, "apply", _fake_triton_apply)

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
        # BF16 layers dispatch unquantized rows without scales.
        recv_scale = (
            torch.randint(
                0, 256, (capacity, hidden // 32), generator=gen, dtype=torch.uint8
            )
            if backend == "mxfp4"
            else None
        )
        for j, (s, t) in enumerate(received):
            recv_x[j], recv_ids[j], recv_w[j] = x[s][t], ids[s][t], w[s][t]
        if backend == "mxfp4":
            # AITER takes a 0/1 mask with a trailing always-masked slot.
            expert_map = torch.zeros(num_experts + 1, dtype=torch.int32)
            expert_map[list(owned)] = 1
        else:
            expert_map = torch.full((num_experts,), -1, dtype=torch.int32)
            expert_map[list(owned)] = torch.arange(local, dtype=torch.int32)

        experts = _experts(config, norm_fn, backend)
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
            expert_map,
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
        if backend == "mxfp4":
            torch.testing.assert_close(
                call["hidden_states"], recv_x.repeat_interleave(top_k, dim=0)
            )
            torch.testing.assert_close(
                call["a1q_scale"], recv_scale.repeat_interleave(top_k, dim=0)
            )
            assert int(call["num_local_tokens"]) == n * top_k
        else:
            # Stale rows reach the Triton GEMMs only as padding routes.
            assert bool((call["topk_ids"][n:] == -1).all())
            torch.testing.assert_close(call["topk_ids"][:n], recv_ids[:n])
            assert call["a1q_scale"] is None
        assert experts._take_route_output() is None
        # Every row, including the stale tail, is finite.
        assert bool(torch.isfinite(output).all())
        # MoRI combine: an unweighted sum of the returned rows per source token.
        for j, (s, t) in enumerate(received):
            combined[s][t] += output[j]
    return combined, reference


@pytest.mark.parametrize("backend", ["mxfp4", "bf16"])
@pytest.mark.parametrize(
    "ep_size,num_experts,top_k", [(2, 8, 2), (4, 16, 4), (8, 64, 6)]
)
def test_ep_rank_partials_sum_to_the_dp1_output(
    monkeypatch: pytest.MonkeyPatch,
    ep_size: int,
    num_experts: int,
    top_k: int,
    backend: str,
) -> None:
    combined, reference = _simulate_mori_ep(
        monkeypatch,
        ep_size=ep_size,
        num_experts=num_experts,
        top_k=top_k,
        hidden=64,
        tokens_per_rank=5,
        seed=ep_size,
        backend=backend,
    )
    for got, ref in zip(combined, reference):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


def _fake_quark_init(self, moe, weight_quant_key, activation_quant_key) -> None:
    self.moe = moe
    self.ocp_mx_scheme = "w_mxfp4_a_mxfp4"
    self.mxfp4_backend = Mxfp4MoeBackend.AITER_MXFP4_MXFP4
    self.experts_cls = None


def test_mixed_bf16_and_mxfp4_methods_construct_under_mori(
    monkeypatch: pytest.MonkeyPatch, default_vllm_config
) -> None:
    """Layers 0 and 61 are BF16 (Triton), the rest MXFP4 (AITER)."""
    monkeypatch.setattr(QuarkOCP_MX_MoEMethod, "__init__", _fake_quark_init)
    mori = _moe_config(dp_size=8, num_experts=128)
    bf16_method = MuseSparkUnquantizedFusedMoE(mori)
    mxfp4_method = MuseSparkQuarkOCPMXFusedMoE(mori, kMxfp4Static, kMxfp4Dynamic)
    assert bf16_method.experts_cls is MuseSparkTritonExperts
    assert mxfp4_method.experts_cls is MuseSparkAiterExperts


def test_mixed_bf16_and_mxfp4_layers_under_mori(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both expert kinds finalize their local routes, and only MXFP4 expands
    its input per route -- after dispatch, never before it."""
    mori = _moe_config(dp_size=8, num_experts=128)
    norm = _rms_norm_with_input_scale(torch.ones(64), 1e-6)
    for backend in ("bf16", "mxfp4"):
        experts = _experts(mori, norm, backend)
        assert experts._local_finalize is not None
        assert isinstance(
            experts.finalize_weight_and_reduce_impl(), TopKWeightAndReduceNoOP
        )
        shapes = experts.workspace_shapes(
            7, 128, 64, 4, 128, 16, None, MoEActivation.SILU
        )
        assert shapes[2] == (7, 64)

    def _takes_expanded(method) -> bool:
        moe = SimpleNamespace(
            pre_expert_norm=SimpleNamespace(variance_size_override=None),
            experts=SimpleNamespace(
                routed_experts=SimpleNamespace(quant_method=method)
            ),
        )
        return MuseSparkMoE._experts_take_expanded_input.func(moe)

    bf16_method = MuseSparkUnquantizedFusedMoE.__new__(MuseSparkUnquantizedFusedMoE)
    bf16_method.moe = mori
    mxfp4_method = MuseSparkQuarkOCPMXFusedMoE.__new__(MuseSparkQuarkOCPMXFusedMoE)
    mxfp4_method.moe = mori
    assert not _takes_expanded(bf16_method)
    assert not _takes_expanded(mxfp4_method)
    mxfp4_method.moe = _moe_config(dp_size=1, use_ep=False)
    assert _takes_expanded(mxfp4_method)
