# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Muse Spark ROCm MXFP4 under DP + EP with the MoRI EPv1 all2all.

MoRI sends a token once to every rank owning one of its experts and its combine
only sums the rows those ranks return, so each expert rank must finalize its
own routes: ``sum_k w_k * norm(out_k)`` over its local experts.
"""

from collections.abc import Callable
from dataclasses import fields
from math import prod

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.models.muse_spark.amd.muse_moe_runner as mori_runner
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
from vllm.models.muse_spark.amd.ops.gather_rows import gather_rows
from vllm.models.muse_spark.amd.muse_moe_runner import (
    MuseSparkAiterExperts,
    MuseSparkQuarkOCPMXFusedMoE,
    MuseSparkQuarkOCPMXMoEQuantConfig,
    MuseSparkTopKWeightAndReduce,
    MuseSparkTritonExperts,
    MuseSparkUnquantizedFusedMoE,
    MuseSparkUnquantizedMoEQuantConfig,
    finalizes_before_mori_combine,
    route_chunk_tokens,
    validate_bf16_deployment,
    validate_fp8_deployment,
    validate_mxfp4_deployment,
)
from vllm.models.muse_spark.common.ops.fused_finalize import (
    fused_post_expert_finalize,
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
    routing: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run the Muse experts' apply on every simulated EP rank behind a
    MoRI-style dispatch/combine. Returns (combined, reference) per source rank.

    ``mxfp4`` runs MuseSparkAiterExperts with AITER faked; ``bf16`` runs
    MuseSparkTritonExperts with the stock Triton GEMMs faked. ``rank0_only``
    routes every token to rank 0's experts, so the other ranks receive nothing.
    """
    gen = torch.Generator().manual_seed(seed)
    local = num_experts // ep_size
    assert top_k <= local
    eps = 1e-6
    gain = torch.rand(hidden, generator=gen) + 0.5
    norm_fn = _rms_norm_with_input_scale(gain, eps)
    expert_w = torch.randn(num_experts, hidden, hidden, generator=gen) / hidden**0.5

    pool = local if routing == "rank0_only" else num_experts
    x = [torch.randn(tokens_per_rank, hidden, generator=gen) for _ in range(ep_size)]
    ids = [
        torch.stack(
            [
                torch.randperm(pool, generator=gen)[:top_k]
                for _ in range(tokens_per_rank)
            ]
        ).to(torch.int32)
        for _ in range(ep_size)
    ]
    # Token 0 of rank 0 routes only to rank 0's experts.
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
        # One output row per packed route, within the per-call byte budget.
        assert hs.shape[0] * hidden * 4 <= mori_runner._ROUTE_CHUNK_BYTES
        # Packed slots past the route count carry the always-masked expert.
        assert bool((flat_ids[n:, 0] == mask.shape[0] - 1).all())
        calls.append({k: kwargs[k] for k in ("hidden_states", "a1q_scale")})
        calls[-1].update(ids=flat_ids[:n, 0].clone())
        # Rows AITER skips (past the route count) are left unwritten.
        out = torch.full((hs.shape[0], hidden), float("nan"))
        for i in range(n):
            e = int(flat_ids[i, 0])
            assert mask[e]
            out[i] = hs[i].float() @ expert_w[e]
        return out.to(kwargs["output_dtype"])

    monkeypatch.setattr(mori_runner, "rocm_aiter_fused_experts", _fake_aiter)

    def _fake_triton_apply(
        self, output, hidden_states, w1, w2, topk_weights, topk_ids, activation,
        global_num_experts, expert_map, a1q_scale, a2_scale, workspace13,
        workspace2, expert_tokens_meta, apply_router_weight_on_input,
    ) -> None:  # fmt: skip
        assert not self._apply_router_weight_in_gemm2(apply_router_weight_on_input)
        assert output.shape == hidden_states.shape
        calls.append({"ids": topk_ids.clone(), "a1q_scale": a1q_scale})
        # The GEMMs use workspace13 as scratch.
        workspace13.fill_(float("nan"))
        # GEMM2 rows of skipped routes (remote, padding) are never written.
        rows = torch.full((*topk_ids.shape, hidden), float("nan"))
        for i, k in (topk_ids >= 0).nonzero().tolist():
            e = int(topk_ids[i, k])
            if expert_map[e] >= 0:
                rows[i, k] = hidden_states[i].float() @ expert_w[e]
        self._finalize_expert_output(rows, output, topk_ids, expert_map)

    monkeypatch.setattr(TritonExperts, "apply", _fake_triton_apply)
    # A small budget so several chunks run per rank.
    monkeypatch.setattr(mori_runner, "_ROUTE_CHUNK_BYTES", 3 * top_k * hidden * 4)

    config = _moe_config(
        dp_size=ep_size, num_experts=num_experts, top_k=top_k, hidden_size=hidden
    )
    config.in_dtype = torch.float32
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
        ws13_shape, _, out_shape = experts.workspace_shapes(
            capacity, 2 * hidden, hidden, top_k, num_experts, local, None,
            MoEActivation.SILU,
        )  # fmt: skip
        assert out_shape == (capacity, hidden)
        # As the modular kernel: workspace13 and the output share storage.
        common = torch.full(
            (max(prod(ws13_shape), prod(out_shape)),), float("nan")
        )
        output = common[: prod(out_shape)].view(out_shape)
        workspace13 = common[: prod(ws13_shape)].view(ws13_shape)
        first_call = len(calls)
        experts.apply(
            output,
            recv_x,
            torch.empty(local, 2 * hidden, 0),
            torch.empty(0),
            recv_w,
            recv_ids,
            MoEActivation.SILU,
            num_experts,
            expert_map,
            recv_scale,
            None,
            workspace13,
            torch.empty(0),
            mk.ExpertTokensMetadata(
                expert_num_tokens=torch.tensor([n], dtype=torch.int32),
                expert_num_tokens_cpu=None,
            ),
            False,
        )
        rank_calls = calls[first_call:]
        chunk_tokens = (
            3
            if backend == "mxfp4"
            else experts._chunk_tokens(2 * hidden, hidden, top_k)
        )
        assert len(rank_calls) == -(-capacity // chunk_tokens)
        if backend == "mxfp4":
            # Packed token-major: (received row, local expert) per route.
            packed = [
                (j, int(e)) for j in range(n) for e in recv_ids[j] if int(e) in owned
            ]
            rows = torch.tensor([j for j, _ in packed], dtype=torch.long)
            got_ids = torch.cat([c["ids"] for c in rank_calls])
            torch.testing.assert_close(
                got_ids, torch.tensor([e for _, e in packed], dtype=torch.int32)
            )
            got_x, got_scale = [], []
            for c in rank_calls:
                k = c["ids"].numel()
                got_x.append(c["hidden_states"][:k])
                got_scale.append(c["a1q_scale"][:k])
            torch.testing.assert_close(torch.cat(got_x), recv_x[rows])
            torch.testing.assert_close(torch.cat(got_scale), recv_scale[rows])
        else:
            # Stale rows reach the Triton GEMMs only as padding routes.
            got_ids = torch.cat([c["ids"] for c in rank_calls])
            assert bool((got_ids[n:] == -1).all())
            torch.testing.assert_close(got_ids[:n], recv_ids[:n])
            assert all(c["a1q_scale"] is None for c in rank_calls)
        assert experts._take_route_output() is None
        # Combine reads only the received rows.
        assert bool(torch.isfinite(output[:n]).all())
        # MoRI combine: an unweighted sum of the returned rows per source token.
        for j, (s, t) in enumerate(received):
            combined[s][t] += output[j]
    return combined, reference


@pytest.mark.parametrize("routing", ["random", "rank0_only"])
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
    routing: str,
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
        routing=routing,
    )
    for got, ref in zip(combined, reference):
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)


# Muse Spark: 256 experts, top-8, MoE hidden 8192.
@pytest.mark.parametrize("ep_size", [8, 16, 32])
@pytest.mark.parametrize("max_num_batched_tokens", [2048, 8192])
def test_wide_ep_kernel_buffers_stay_under_32bit_addressing(
    ep_size: int, max_num_batched_tokens: int
) -> None:
    """No buffer handed to AITER or the Triton GEMMs reaches 2**32 bytes, however
    large MoRI's receive capacity (ep_size * max_num_batched_tokens) gets."""
    hidden, top_k, num_experts, intermediate = 8192, 8, 256, 2048
    capacity = ep_size * max_num_batched_tokens
    config = _moe_config(
        dp_size=ep_size, num_experts=num_experts, top_k=top_k, hidden_size=hidden
    )
    limit = 2**32
    routes_per_token = min(top_k, num_experts // ep_size)
    chunk_tokens = route_chunk_tokens(routes_per_token * hidden * 2)
    rows = min(capacity, chunk_tokens) * routes_per_token
    assert rows * hidden * 2 < limit  # AITER route output (bf16)
    assert rows * hidden // 2 < limit  # packed MXFP4 route input
    assert rows * hidden * 2 <= 2**31  # Triton finalize int32 offsets

    norm = _rms_norm_with_input_scale(torch.ones(hidden), 1e-6)
    experts = _experts(config, norm, "bf16")
    ws13, ws2, out = experts.workspace_shapes(
        capacity, 2 * intermediate, hidden, top_k, num_experts,
        num_experts // ep_size, None, MoEActivation.SILU,
    )  # fmt: skip
    assert out == (capacity, hidden)
    assert (prod(ws13) - prod(out)) * 2 <= 2**31
    assert prod(ws2) * 2 <= 2**31


def test_route_chunk_budget_rejects_oversized_tokens() -> None:
    with pytest.raises(ValueError, match="per-kernel buffer budget"):
        route_chunk_tokens(2**31 + 1)


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


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton kernels need a GPU"
)


@requires_gpu
def test_gather_rows_copies_only_counted_rows() -> None:
    src = torch.randint(0, 256, (9, 4096), dtype=torch.uint8, device="cuda")
    index = torch.tensor([4, 0, 8, 4, 1], device="cuda")
    out = gather_rows(src, index, torch.tensor([3], device="cuda"))
    torch.testing.assert_close(out[:3], src[index[:3]])
    cpu = gather_rows(src.cpu(), index.cpu(), torch.tensor([3]))
    torch.testing.assert_close(out[:3].cpu(), cpu[:3])


@requires_gpu
@pytest.mark.parametrize("hidden", [4096, 8192])
def test_permuted_finalize_skips_rows_past_num_tokens(hidden: int) -> None:
    torch.manual_seed(0)
    tokens, top_k = 6, 8
    routes = torch.randn(tokens * top_k, hidden, device="cuda", dtype=torch.bfloat16)
    idx = torch.arange(tokens * top_k, device="cuda", dtype=torch.int32)
    idx = idx.view(tokens, top_k).contiguous()
    idx[:, 1::2] = -1
    weights = torch.rand(tokens, top_k, device="cuda")
    gain = torch.randn(hidden, device="cuda", dtype=torch.bfloat16) / 10
    sentinel = torch.full((tokens, hidden), 7.0, device="cuda", dtype=torch.bfloat16)
    out = sentinel.clone()
    fused_post_expert_finalize(
        routes, gain, weights, 1e-6, 1.0, True, out,
        permuted_idx=idx, num_tokens=torch.tensor([4], device="cuda"),
    )  # fmt: skip
    full = sentinel.clone()
    fused_post_expert_finalize(
        routes, gain, weights, 1e-6, 1.0, True, full, permuted_idx=idx
    )
    torch.testing.assert_close(out[:4], full[:4])
    torch.testing.assert_close(out[4:], sentinel[4:])
