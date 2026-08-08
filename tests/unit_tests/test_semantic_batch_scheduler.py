import concurrent.futures
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import BatchFeature

from rlinf.envs.libero.libero_env import LiberoEnv
from rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model import (
    FlowMatchingActionHeadForRLActionPrediction,
    GR00T_N1_7_ForRLActionPrediction,
    _dit_cross_attention_trainable_prefixes,
    _dit_tail_trainable_prefixes,
    _prepare_action_only_observation,
    _resize_semantic_token_axis,
    _semantic_publish_due,
    _stale_age_gate,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_preprocess_proxy import (
    Gr00tN1d7SemanticPreprocessProxy,
)
from rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server import (
    Gr00tN1d7RawObservationPublisher,
    Gr00tN1d7SemanticBackbonePolicy,
    Gr00tN1d7SemanticCacheClient,
    _ZmqRpcServer,
    batch_feature_to_payload,
    dequantize_semantic_transport,
    payload_to_batch_feature,
    quantize_semantic_transport,
)
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.workers.env.env_worker import (
    EnvWorker,
    _resolve_semantic_eval_publish_frame,
    _staggered_semantic_frame,
    _validate_semantic_publish_frame,
)


@pytest.mark.parametrize("source_tokens", [128, 570])
def test_resize_semantic_token_axis_stabilizes_ppo_replay(source_tokens):
    outputs = BatchFeature(
        data={
            "backbone_features": torch.arange(
                2 * source_tokens * 3, dtype=torch.float32
            ).reshape(2, source_tokens, 3),
            "backbone_attention_mask": torch.ones(2, source_tokens, dtype=torch.bool),
            "image_mask": torch.ones(2, dtype=torch.bool),
        }
    )

    resized = _resize_semantic_token_axis(outputs, 160)

    assert resized["backbone_features"].shape == (2, 160, 3)
    assert resized["backbone_attention_mask"].shape == (2, 160)
    assert resized["image_mask"].shape == (2,)
    preserved = min(source_tokens, 160)
    torch.testing.assert_close(
        resized["backbone_features"][:, :preserved],
        outputs["backbone_features"][:, :preserved],
    )
    if source_tokens < 160:
        assert not resized["backbone_attention_mask"][:, source_tokens:].any()


class _PendingPolicy:
    def __init__(
        self, ready: int, total: int, ready_envs: int = 0, total_envs: int = 0
    ):
        self._counts = (ready, total)
        self._env_counts = (ready_envs, total_envs)
        self.urgent = False
        self._lock = threading.Lock()

    def pending_request_counts(self) -> tuple[int, int]:
        with self._lock:
            return self._counts

    def pending_env_counts(self) -> tuple[int, int]:
        with self._lock:
            return self._env_counts

    def freshness_demand_active(self) -> bool:
        return self.urgent

    def set_counts(self, ready: int, total: int) -> None:
        with self._lock:
            self._counts = (ready, total)

    def set_env_counts(self, ready: int, total: int) -> None:
        with self._lock:
            self._env_counts = (ready, total)


def _server(
    policy: _PendingPolicy,
    target: int,
    wait_ms: float,
    target_envs: int = 0,
    bootstrap_target_envs: int = 0,
    bootstrap_wait_ms: float = 30000.0,
) -> _ZmqRpcServer:
    server = object.__new__(_ZmqRpcServer)
    server.policy = policy
    server.batch_target_requests = target
    server.batch_target_envs = target_envs
    server.batch_wait_ms = wait_ms
    server.bootstrap_target_envs = bootstrap_target_envs
    server.bootstrap_wait_ms = bootstrap_wait_ms
    server._bootstrap_complete = bootstrap_target_envs <= 0
    server._bootstrap_deadline_perf = None
    server.running = True
    server._scheduler_wakeup = threading.Event()
    return server


def test_dit_tail_trainable_prefixes_select_only_requested_tail_blocks():
    action_head = SimpleNamespace(
        model=SimpleNamespace(transformer_blocks=[object() for _ in range(8)])
    )

    prefixes = _dit_tail_trainable_prefixes(action_head, 3)

    assert "action_head.model.transformer_blocks.4" not in prefixes
    assert "action_head.model.transformer_blocks.5" in prefixes
    assert "action_head.model.transformer_blocks.7" in prefixes
    assert "action_head.model.proj_out_2" in prefixes
    assert "action_head.stale_residual_adapter" not in prefixes
    assert "action_head.stale_semantic_token_adapter" not in prefixes
    assert "action_head.value_head" not in prefixes
    assert "action_head.state_encoder" not in prefixes
    assert "action_head.action_encoder" not in prefixes
    assert "action_head.action_decoder" not in prefixes


def test_dit_tail_trainable_prefixes_reject_invalid_block_count():
    action_head = SimpleNamespace(
        model=SimpleNamespace(transformer_blocks=[object() for _ in range(4)])
    )

    with pytest.raises(ValueError, match="must be in"):
        _dit_tail_trainable_prefixes(action_head, 5)


def test_dit_cross_attention_prefixes_select_semantic_projections_only():
    action_head = SimpleNamespace(
        model=SimpleNamespace(
            transformer_blocks=[
                SimpleNamespace(cross_attention_dim=2048),
                SimpleNamespace(cross_attention_dim=None),
                SimpleNamespace(cross_attention_dim=2048),
            ]
        )
    )

    prefixes = _dit_cross_attention_trainable_prefixes(action_head)

    assert "action_head.model.transformer_blocks.0.attn1.to_k" in prefixes
    assert "action_head.model.transformer_blocks.2.attn1.to_v" in prefixes
    assert "action_head.model.transformer_blocks.2.attn1.to_out" in prefixes
    assert "action_head.model.transformer_blocks.0.attn1.to_q" not in prefixes
    assert not any("transformer_blocks.1" in prefix for prefix in prefixes)
    assert "action_head.model.proj_out_2" in prefixes
    assert not any(".ff" in prefix for prefix in prefixes)


def test_dit_cross_attention_prefixes_can_include_queries():
    action_head = SimpleNamespace(
        model=SimpleNamespace(
            transformer_blocks=[SimpleNamespace(cross_attention_dim=2048)]
        )
    )

    prefixes = _dit_cross_attention_trainable_prefixes(action_head, include_query=True)

    assert "action_head.model.transformer_blocks.0.attn1.to_q" in prefixes


def test_bootstrap_waits_past_steady_state_window_until_all_envs_ready():
    policy = _PendingPolicy(ready=1, total=1, ready_envs=5, total_envs=5)
    server = _server(
        policy,
        target=0,
        wait_ms=5,
        target_envs=120,
        bootstrap_target_envs=10,
        bootstrap_wait_ms=500,
    )

    def finish_bootstrap():
        time.sleep(0.03)
        policy.set_counts(ready=2, total=2)
        policy.set_env_counts(ready=10, total=10)
        server._scheduler_wakeup.set()

    thread = threading.Thread(target=finish_bootstrap)
    thread.start()
    started = time.perf_counter()
    assert server._wait_for_ready_batch() == 2
    elapsed = time.perf_counter() - started
    thread.join()
    assert elapsed >= 0.02
    assert server._bootstrap_complete


def test_ready_target_releases_batch_before_timeout():
    policy = _PendingPolicy(ready=1, total=4)
    server = _server(policy, target=4, wait_ms=500)

    def finish_batch():
        time.sleep(0.02)
        policy.set_counts(ready=4, total=4)
        server._scheduler_wakeup.set()

    thread = threading.Thread(target=finish_batch)
    thread.start()
    started = time.perf_counter()
    assert server._wait_for_ready_batch() == 4
    elapsed = time.perf_counter() - started
    thread.join()
    assert elapsed < 0.25


def test_timeout_releases_partial_ready_batch():
    policy = _PendingPolicy(ready=2, total=2)
    server = _server(policy, target=4, wait_ms=20)

    started = time.perf_counter()
    assert server._wait_for_ready_batch() == 2
    elapsed = time.perf_counter() - started
    assert elapsed >= 0.015


def test_timeout_does_not_drop_unfinished_preprocessing():
    policy = _PendingPolicy(ready=0, total=4)
    server = _server(policy, target=4, wait_ms=10)

    def finish_one_request():
        time.sleep(0.03)
        policy.set_counts(ready=1, total=4)
        server._scheduler_wakeup.set()

    thread = threading.Thread(target=finish_one_request)
    thread.start()
    assert server._wait_for_ready_batch() == 1
    thread.join()


def test_freshness_demand_preserves_short_coalescing_window():
    policy = _PendingPolicy(ready=3, total=16, ready_envs=12, total_envs=64)
    policy.urgent = True
    server = _server(policy, target=0, target_envs=64, wait_ms=500)

    def finish_env_batch():
        time.sleep(0.02)
        policy.set_counts(ready=16, total=16)
        policy.set_env_counts(ready=64, total=64)
        server._scheduler_wakeup.set()

    thread = threading.Thread(target=finish_env_batch)
    thread.start()
    started = time.perf_counter()
    assert server._wait_for_ready_batch() == 16
    elapsed = time.perf_counter() - started
    thread.join()
    assert elapsed >= 0.015
    assert elapsed < 0.25


def test_boundary_packet_priority_precedes_urgent_mid_chunk_packet():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy._freshness_requirements = {1: (0, 16, 1)}
    policy.semantic_cache_by_env = {}
    boundary = {
        "priority": 0,
        "env_ids": [1],
        "published_wallclock_s": 2.0,
    }
    mid_chunk = {
        "priority": 1,
        "env_ids": [1],
        "published_wallclock_s": 1.0,
    }

    assert policy._pending_packet_priority(boundary) < policy._pending_packet_priority(
        mid_chunk
    )


def test_env_target_releases_complete_env_batch():
    policy = _PendingPolicy(ready=8, total=16, ready_envs=32, total_envs=64)
    server = _server(policy, target=0, target_envs=64, wait_ms=500)

    def finish_env_batch():
        time.sleep(0.02)
        policy.set_counts(ready=16, total=16)
        policy.set_env_counts(ready=64, total=64)
        server._scheduler_wakeup.set()

    thread = threading.Thread(target=finish_env_batch)
    thread.start()
    assert server._wait_for_ready_batch() == 16
    thread.join()


def test_fetch_waits_only_until_target_age_is_available():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy.fetch_pause_s = 0.0
    policy.last_scheduler_error = None
    policy._freshness_demand = threading.Event()
    wakeup = threading.Event()
    policy.scheduler_wakeup_callback = wakeup.set
    policy._cache_lock = threading.RLock()
    policy.pending_batches = {}

    def packet(frame_id: int):
        return {
            "backbone_output": {"backbone_features": torch.zeros(1, 2, 3)},
            "episode_generation": 0,
            "source_frame_id": frame_id,
            "source_wallclock_s": time.time(),
            "completed_wallclock_s": time.time(),
            "version": frame_id + 1,
            "forward_ms": 1.0,
            "batch_size": 1,
        }

    initial = packet(0)
    policy.semantic_cache_history_by_env = {1: deque([initial], maxlen=8)}
    policy.semantic_cache_by_env = {1: initial}

    def publish_fresh_packet():
        time.sleep(0.02)
        fresh = packet(8)
        with policy._cache_lock:
            policy.semantic_cache_history_by_env[1].append(fresh)
            policy.semantic_cache_by_env[1] = fresh

    thread = threading.Thread(target=publish_fresh_packet)
    thread.start()
    response = policy.fetch_latest(
        {
            "env_ids": [1],
            "episode_generations": [0],
            "current_frame_ids": [16],
            "target_age_frames": 8,
            "max_wait_ms": 200,
        }
    )
    thread.join()

    assert response["metadata"]["source_frame_ids"] == [8]
    assert wakeup.is_set()
    assert response["metrics"]["semantic_server_freshness_target_met"] == 1.0
    assert 10 <= response["metrics"]["semantic_server_freshness_wait_ms"] < 150


def test_missing_zero_wait_semantic_requests_urgent_flush():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy.fetch_pause_s = 0.0
    policy.last_scheduler_error = None
    policy._freshness_demand = threading.Event()
    wakeup = threading.Event()
    policy.scheduler_wakeup_callback = wakeup.set
    policy._cache_lock = threading.RLock()
    policy.pending_batches = {}
    policy.semantic_cache_history_by_env = {}
    policy.semantic_cache_by_env = {}

    response = policy.fetch_latest(
        {
            "env_ids": [3],
            "episode_generations": [1],
            "current_frame_ids": [0],
            "target_age_frames": 8,
            "max_wait_ms": 0,
        }
    )

    assert response == {"ready": False, "missing_env_ids": [3]}
    assert wakeup.is_set()
    assert policy.freshness_demand_active()


def test_fetch_exact_waits_for_matching_frame_and_rejects_nearest_packet():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy.last_scheduler_error = None
    policy._freshness_demand = threading.Event()
    policy._freshness_requirements = {}
    policy.scheduler_wakeup_callback = None
    policy._cache_lock = threading.RLock()

    def packet(frame_id: int):
        return {
            "backbone_output": {
                "backbone_features": torch.full((1, 2, 3), float(frame_id))
            },
            "episode_generation": 2,
            "source_frame_id": frame_id,
            "source_wallclock_s": time.time(),
            "completed_wallclock_s": time.time(),
            "version": frame_id + 1,
            "forward_ms": 1.0,
            "batch_size": 1,
        }

    stale = packet(8)
    policy.semantic_cache_history_by_env = {4: deque([stale], maxlen=32)}
    policy.semantic_cache_by_env = {4: stale}

    missing = policy.fetch_exact(
        {
            "env_ids": [4],
            "episode_generations": [2],
            "source_frame_ids": [16],
            "max_wait_ms": 0,
        }
    )
    assert missing == {"ready": False, "missing_env_ids": [4]}

    def publish_exact_packet():
        time.sleep(0.02)
        with policy._cache_lock:
            policy.semantic_cache_history_by_env[4].append(packet(16))

    thread = threading.Thread(target=publish_exact_packet)
    thread.start()
    response = policy.fetch_exact(
        {
            "env_ids": [4],
            "episode_generations": [2],
            "source_frame_ids": [16],
            "max_wait_ms": 200,
        }
    )
    thread.join()

    restored = payload_to_batch_feature(response["backbone_outputs"])
    assert response["metadata"]["source_frame_ids"] == [16]
    assert restored["backbone_features"].unique().item() == 16


def test_semantic_history_retains_previous_episode_for_ppo_replay():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy.cache_history_size = 8
    policy.last_scheduler_error = None
    policy._freshness_demand = threading.Event()
    policy._freshness_requirements = {}
    policy.scheduler_wakeup_callback = None
    policy._cache_lock = threading.RLock()
    policy.semantic_cache_history_by_env = {}
    policy.semantic_cache_by_env = {}

    def packet(generation: int, frame_id: int):
        return {
            "backbone_output": {
                "backbone_features": torch.full(
                    (1, 2, 3), float(generation * 100 + frame_id)
                )
            },
            "episode_generation": generation,
            "source_frame_id": frame_id,
            "source_wallclock_s": time.time(),
            "completed_wallclock_s": time.time(),
            "version": generation + 1,
            "forward_ms": 1.0,
            "batch_size": 1,
        }

    previous = packet(2, 16)
    current = packet(3, 0)
    policy._store_semantic_cache_entry(4, previous)
    policy._store_semantic_cache_entry(4, current)

    response = policy.fetch_exact(
        {
            "env_ids": [4],
            "episode_generations": [2],
            "source_frame_ids": [16],
            "max_wait_ms": 0,
        }
    )

    restored = payload_to_batch_feature(response["backbone_outputs"])
    assert response["metadata"]["episode_generations"] == [2]
    assert restored["backbone_features"].unique().item() == 216
    assert policy.semantic_cache_by_env[4] is current


def test_rpc_server_fetch_waits_overlap():
    barrier = threading.Barrier(2)

    class Policy:
        def fetch_latest(self, data):
            barrier.wait(timeout=0.5)
            time.sleep(0.01)
            return {"request_id": data["request_id"]}

    server = object.__new__(_ZmqRpcServer)
    server.policy = Policy()
    server._fetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    server._fetch_futures = {}
    responses = []
    server._send = lambda envelope, result: responses.append((envelope, result))

    try:
        server._submit_fetch([b"first"], "fetch_latest", {"request_id": 1})
        server._submit_fetch([b"second"], "fetch_latest", {"request_id": 2})
        deadline = time.perf_counter() + 1.0
        while len(responses) < 2 and time.perf_counter() < deadline:
            server._drain_fetch_responses()
            time.sleep(0.001)
    finally:
        server._fetch_executor.shutdown(wait=True, cancel_futures=True)

    assert {response["request_id"] for _, response in responses} == {1, 2}


def test_semantic_cache_client_fetches_shards_in_parallel_and_restores_order(
    monkeypatch,
):
    client = Gr00tN1d7SemanticCacheClient(
        host="127.0.0.1,127.0.0.1",
        port="6666,6668",
    )
    rendezvous = threading.Barrier(2, timeout=0.5)
    worker_threads = set()

    def fetch_shard(shard, request):
        worker_threads.add(threading.get_ident())
        rendezvous.wait()
        env_ids = request["env_ids"]
        return {
            "ready": True,
            "backbone_outputs": batch_feature_to_payload(
                {"backbone_features": torch.tensor(env_ids).reshape(-1, 1, 1)}
            ),
            "metadata": {"source_frame_ids": [env_id + 10 for env_id in env_ids]},
            "metrics": {},
        }

    monkeypatch.setattr(client, "_call_fetch_shard", fetch_shard)
    try:
        outputs, metadata = client.fetch_latest(
            env_ids=[1, 0],
            episode_generations=[0, 0],
            current_frame_ids=[16, 16],
            wait_for_initial=False,
            device="cpu",
            floating_dtype=torch.float32,
        )
    finally:
        client.close()

    assert len(worker_threads) == 2
    assert outputs["backbone_features"].flatten().tolist() == [1.0, 0.0]
    assert metadata["source_frame_ids"] == [11, 10]


def test_semantic_cache_client_background_fetch_is_latest_only(monkeypatch):
    client = Gr00tN1d7SemanticCacheClient(host="127.0.0.1", port=6666)
    release = threading.Event()
    calls = []

    def fetch_response(**request):
        frame_id = request["current_frame_ids"][0]
        calls.append(frame_id)
        if frame_id >= 8:
            assert release.wait(timeout=1.0)
        return (
            BatchFeature(
                data={"backbone_features": torch.full((1, 2, 3), float(frame_id))}
            ),
            {
                "source_frame_ids": [frame_id],
                "episode_generations": [0],
                "semantic_versions": [frame_id + 1],
            },
            {"semantic_cache_fetch_ms": 1.0},
        )

    monkeypatch.setattr(client, "_fetch_latest_response", fetch_response)
    try:
        client.submit_latest(
            env_ids=[1],
            episode_generations=[0],
            current_frame_ids=[0],
            floating_dtype=torch.float32,
        )
        initial = client.wait_latest(device="cpu", floating_dtype=torch.float32)
        assert initial is not None
        assert initial[0]["backbone_features"].unique().item() == 0

        client.submit_latest(
            env_ids=[1],
            episode_generations=[0],
            current_frame_ids=[8],
            floating_dtype=torch.float32,
        )
        client.submit_latest(
            env_ids=[1],
            episode_generations=[0],
            current_frame_ids=[12],
            floating_dtype=torch.float32,
        )
        client.submit_latest(
            env_ids=[1],
            episode_generations=[0],
            current_frame_ids=[16],
            floating_dtype=torch.float32,
        )
        started = time.perf_counter()
        assert client.poll_latest(device="cpu", floating_dtype=torch.float32) is None
        assert time.perf_counter() - started < 0.05

        release.set()
        first = client.wait_latest(device="cpu", floating_dtype=torch.float32)
        latest = client.wait_latest(device="cpu", floating_dtype=torch.float32)
    finally:
        release.set()
        client.close()

    assert first is not None and latest is not None
    assert first[0]["backbone_features"].unique().item() == 8
    assert latest[0]["backbone_features"].unique().item() == 16
    assert calls == [0, 8, 16]
    assert client.last_metrics["semantic_cache_fetch_replaced"] == 2.0


def test_semantic_cache_client_wait_timeout_preserves_inflight_fetch(monkeypatch):
    client = Gr00tN1d7SemanticCacheClient(host="127.0.0.1", port=6666)
    release = threading.Event()

    def fetch_response(**request):
        assert release.wait(timeout=1.0)
        return (
            BatchFeature(data={"backbone_features": torch.ones(1, 2, 3)}),
            {
                "source_frame_ids": request["current_frame_ids"],
                "episode_generations": request["episode_generations"],
            },
            {"semantic_cache_fetch_ms": 1.0},
        )

    monkeypatch.setattr(client, "_fetch_latest_response", fetch_response)
    try:
        client.submit_latest(
            env_ids=[1],
            episode_generations=[0],
            current_frame_ids=[8],
            floating_dtype=torch.float32,
        )
        started = time.perf_counter()
        timed_out = client.wait_latest(
            device="cpu", floating_dtype=torch.float32, timeout_ms=10.0
        )
        assert timed_out is None
        assert time.perf_counter() - started < 0.1
        assert client.last_metrics["semantic_cache_foreground_timeout"] == 1.0

        release.set()
        completed = client.wait_latest(
            device="cpu", floating_dtype=torch.float32, timeout_ms=1000.0
        )
    finally:
        release.set()
        client.close()

    assert completed is not None
    assert completed[1]["source_frame_ids"] == [8]


def test_semantic_cache_client_speculative_fetch_overrides_wait(monkeypatch):
    client = Gr00tN1d7SemanticCacheClient(
        host="127.0.0.1", port=6666, fetch_max_wait_ms=120000
    )
    requests = []

    def fetch_ready(shard, request):
        requests.append((shard, request))
        return {
            "ready": True,
            "backbone_outputs": batch_feature_to_payload(
                {"backbone_features": torch.zeros(1, 2, 3)}
            ),
            "metadata": {
                "env_ids": [1],
                "source_frame_ids": [16],
                "episode_generations": [2],
                "semantic_versions": [3],
            },
            "metrics": {},
        }

    monkeypatch.setattr(client, "_call_fetch_shard", fetch_ready)
    try:
        client.submit_latest(
            env_ids=[1],
            episode_generations=[2],
            current_frame_ids=[32],
            floating_dtype=torch.float32,
            max_wait_ms=0.0,
        )
        assert client.wait_latest(device="cpu", floating_dtype=torch.float32)
    finally:
        client.close()

    assert len(requests) == 1
    assert requests[0][1]["max_wait_ms"] == 0.0


def test_semantic_cache_client_background_missing_env_completes_without_retry(
    monkeypatch,
):
    client = Gr00tN1d7SemanticCacheClient(host="127.0.0.1", port=6666)
    requests = []

    def fetch_missing(shard, request):
        requests.append((shard, request))
        return {"ready": False, "missing_env_ids": request["env_ids"]}

    monkeypatch.setattr(client, "_call_fetch_shard", fetch_missing)
    try:
        client.submit_latest(
            env_ids=[1],
            episode_generations=[2],
            current_frame_ids=[32],
            floating_dtype=torch.float32,
        )
        assert client.wait_latest(device="cpu", floating_dtype=torch.float32) is None
    finally:
        client.close()

    assert len(requests) == 1
    assert client.last_metrics["semantic_cache_ready"] == 0.0


def test_raw_publish_replaces_pending_preprocess_future():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy._cache_lock = threading.RLock()
    policy.pending_raw_batches = {}
    policy.scheduler_wakeup_callback = None
    policy._raw_preprocess_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )
    policy._prepare_raw_observation = lambda observation: observation

    first = {
        "observation": {"states": "old"},
        "metadata": {"env_ids": [1], "frame_ids": [8]},
    }
    second = {
        "observation": {"states": "new"},
        "metadata": {"env_ids": [1], "frame_ids": [24]},
    }

    try:
        policy.publish_raw_observations(first)
        response = policy.publish_raw_observations(second)
        packet = policy.pending_raw_batches[(1,)]

        assert response["pending_raw_batches"] == 1
        assert packet["observation"]["states"] == "new"
        assert packet["preprocess_future"].result()["states"] == "new"
    finally:
        policy._raw_preprocess_executor.shutdown(wait=True)


def test_raw_observation_publisher_drains_latest_queue_without_poll(monkeypatch):
    publisher = Gr00tN1d7RawObservationPublisher(host="127.0.0.1", port=6666)
    first_started = threading.Event()
    release_first = threading.Event()
    latest_finished = threading.Event()
    calls = []

    def publish_worker(observation, metadata):
        del observation
        frame_id = metadata["frame_ids"][0]
        calls.append(frame_id)
        if frame_id == 8:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        if frame_id == 24:
            latest_finished.set()
        return []

    monkeypatch.setattr(publisher, "_worker", publish_worker)
    try:
        observation = {"states": torch.zeros(1, 1)}
        publisher.publish(observation, {"env_ids": [1], "frame_ids": [8]})
        assert first_started.wait(timeout=1.0)
        publisher.publish(observation, {"env_ids": [1], "frame_ids": [16]})
        publisher.publish(observation, {"env_ids": [1], "frame_ids": [24]})
        release_first.set()

        assert latest_finished.wait(timeout=1.0)
    finally:
        release_first.set()
        publisher.close()

    assert calls == [8, 24]
    assert publisher.replaced_count == 2


def test_raw_observation_publisher_sends_shards_concurrently(monkeypatch):
    both_started = threading.Event()
    lock = threading.Lock()
    started_ports = set()

    class Client:
        def __init__(self, *, port, **kwargs):
            del kwargs
            self.port = port

        def call_endpoint(self, endpoint, payload):
            assert endpoint == "publish_raw_observations"
            assert payload["metadata"]["env_ids"]
            with lock:
                started_ports.add(self.port)
                if len(started_ports) == 2:
                    both_started.set()
            assert both_started.wait(timeout=1.0)
            return {"port": self.port}

        def close(self):
            pass

    monkeypatch.setattr(
        "rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server._ZmqRpcClient",
        Client,
    )
    publisher = Gr00tN1d7RawObservationPublisher(host="127.0.0.1", port="6666,6667")
    try:
        responses = publisher._worker(
            {"states": torch.arange(4).reshape(4, 1)},
            {"env_ids": [0, 1, 2, 3], "frame_ids": [8, 8, 8, 8]},
        )
    finally:
        publisher.close()

    assert started_ports == {6666, 6667}
    assert {response["port"] for response in responses} == {6666, 6667}


def test_raw_preprocess_future_controls_scheduler_readiness():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy._cache_lock = threading.RLock()
    policy.pending_batches = {}
    policy.pending_raw_batches = {}
    policy.scheduler_wakeup_callback = None
    policy._raw_preprocess_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1
    )
    release = threading.Event()

    def prepare(observation):
        release.wait(timeout=1.0)
        return observation

    policy._prepare_raw_observation = prepare
    try:
        policy.publish_raw_observations(
            {
                "observation": {"states": "frame"},
                "metadata": {"env_ids": [1], "frame_ids": [8]},
            }
        )
        assert policy.pending_request_counts() == (0, 1)
        assert policy.pending_env_counts() == (0, 1)

        release.set()
        policy.pending_raw_batches[(1,)]["preprocess_future"].result(timeout=1.0)
        assert policy.pending_request_counts() == (1, 1)
        assert policy.pending_env_counts() == (1, 1)
    finally:
        release.set()
        policy._raw_preprocess_executor.shutdown(wait=True)


def test_successful_fetch_preserves_other_env_freshness_requirement():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy.fetch_pause_s = 0.0
    policy.last_scheduler_error = None
    policy._freshness_demand = threading.Event()
    policy._freshness_requirements = {}
    policy.scheduler_wakeup_callback = None
    policy._cache_lock = threading.RLock()
    policy.pending_batches = {}
    packet = {
        "backbone_output": {"backbone_features": torch.zeros(1, 2, 3)},
        "episode_generation": 0,
        "source_frame_id": 8,
        "source_wallclock_s": time.time(),
        "completed_wallclock_s": time.time(),
        "version": 1,
        "forward_ms": 1.0,
        "batch_size": 1,
    }
    policy.semantic_cache_history_by_env = {2: deque([packet], maxlen=8)}
    policy.semantic_cache_by_env = {2: packet}

    missing = policy.fetch_latest(
        {
            "env_ids": [1],
            "episode_generations": [0],
            "current_frame_ids": [8],
            "target_age_frames": 8,
            "max_wait_ms": 0,
        }
    )
    ready = policy.fetch_latest(
        {
            "env_ids": [2],
            "episode_generations": [0],
            "current_frame_ids": [8],
            "target_age_frames": 8,
            "max_wait_ms": 0,
        }
    )

    assert missing == {"ready": False, "missing_env_ids": [1]}
    assert ready["ready"]
    assert 1 in policy._freshness_requirements
    assert policy.freshness_demand_active()


def test_int8_semantic_transport_roundtrip_halves_payload():
    torch.manual_seed(7)
    features = (torch.randn(3, 7, 16) * 2).to(torch.bfloat16)
    attention_mask = torch.ones(3, 7, dtype=torch.bool)
    encoded = quantize_semantic_transport(
        {"backbone_features": features, "backbone_attention_mask": attention_mask},
        "int8",
    )

    assert encoded["backbone_features"].dtype == torch.int8
    wire = batch_feature_to_payload(encoded)
    restored = dequantize_semantic_transport(
        payload_to_batch_feature(wire, floating_dtype=torch.bfloat16),
        torch.bfloat16,
    )

    error = (restored["backbone_features"].float() - features.float()).abs()
    assert restored["backbone_features"].dtype == torch.bfloat16
    assert torch.equal(restored["backbone_attention_mask"], attention_mask)
    assert error.mean().item() < 0.01
    assert error.max().item() < 0.04
    encoded_bytes = sum(value["data"].nbytes for value in wire.values())
    original_bytes = features.numel() * features.element_size() + attention_mask.numel()
    assert encoded_bytes < original_bytes * 0.6


def test_libero_sparse_chunk_wraps_only_midpoint_and_final_observations():
    env = object.__new__(LiberoEnv)
    env.auto_reset = False
    env.ignore_terminations = False
    wrapped = []

    def step(actions, auto_reset=True, wrap_observation=True):
        frame = len(wrapped) + 1
        wrapped.append(wrap_observation)
        batch_size = actions.shape[0]
        zeros = torch.zeros(batch_size)
        observation = {"frame": frame} if wrap_observation else None
        return observation, zeros, zeros.bool(), zeros.bool(), {}

    env.step = step
    callbacks = []
    obs_list, rewards, terminations, truncations, _ = env.chunk_step(
        np.zeros((2, 16, 7), dtype=np.float32),
        mid_chunk_callback=callbacks.append,
        mid_chunk_frame=8,
        sparse_observations=True,
    )

    assert [index + 1 for index, value in enumerate(wrapped) if value] == [8, 16]
    assert callbacks == [{"frame": 8}]
    assert obs_list[7] == {"frame": 8}
    assert obs_list[-1] == {"frame": 16}
    assert all(value is None for value in obs_list[:7] + obs_list[8:-1])
    assert rewards.shape == terminations.shape == truncations.shape == (2, 16)


def test_libero_reset_clears_elapsed_steps_before_wrapping_observation():
    env = object.__new__(LiberoEnv)
    env.num_envs = 1
    env._is_start = False
    env.is_eval = True
    env.cfg = SimpleNamespace(reset_gripper_open=False)
    env._elapsed_steps = np.array([480], dtype=np.int32)
    env.current_raw_obs = None
    env._reconfigure = lambda reset_state_ids, env_idx: None
    env.env = SimpleNamespace(
        step=lambda actions, env_idx: ([{}], None, np.array([False]), [{}])
    )

    def reset_metrics(env_idx):
        env._elapsed_steps[env_idx] = 0

    wrapped_elapsed_steps = []

    def wrap_obs(raw_obs):
        wrapped_elapsed_steps.append(env._elapsed_steps.copy())
        return {"elapsed_steps": torch.as_tensor(env._elapsed_steps.copy())}

    env._reset_metrics = reset_metrics
    env._wrap_obs = wrap_obs

    observation, _ = env.reset(reset_state_ids=np.array([0]))

    assert wrapped_elapsed_steps[0].tolist() == [0]
    assert observation["elapsed_steps"].tolist() == [0]


def test_central_semantic_cache_bootstraps_current_frame_before_anticipating():
    class Client:
        def __init__(self):
            self.submitted = []
            self.wait_calls = 0

        def poll_latest(self, **kwargs):
            return None

        def submit_latest(self, **kwargs):
            self.submitted.append(kwargs)

        def wait_latest(self, **kwargs):
            self.wait_calls += 1
            if not self.submitted:
                return None
            return (
                BatchFeature(data={"backbone_features": torch.ones(1, 2, 3)}),
                {
                    "env_ids": [7],
                    "source_frame_ids": [0],
                    "episode_generations": [0],
                    "source_wallclock_s": [time.time()],
                    "semantic_versions": [1],
                },
            )

    client = Client()
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter(
        "_device_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    )
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_non_blocking = True
    model._semantic_boundary_publish = False
    model._semantic_boundary_publish_interval = 1
    model._semantic_env_bootstrap_publish = True
    model._semantic_publish_interval_frames = 0
    model._semantic_last_episode_generations = {}
    model._semantic_last_published_frames = {}
    model._semantic_client = client
    model._semantic_cache = BatchFeature(
        data={"backbone_features": torch.full((1, 2, 3), 9.0)}
    )
    model._latest_semantic_metadata = {
        "env_ids": [1_000_000_007],
        "source_frame_ids": [465],
        "episode_generations": [0],
        "source_wallclock_s": [time.time()],
        "semantic_versions": [99],
    }
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([7]),
        "frame_ids": torch.tensor([0]),
        "episode_generations": torch.tensor([0]),
        "observation_wallclock_s": torch.tensor([time.time()]),
    }
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_fetch_hard_max_age_frames = 8
    model.output_action_chunks = 16
    model.compute_dtype = torch.float32

    outputs, age = model._semantic_backbone(BatchFeature(data={}))

    assert outputs["backbone_features"].unique().item() == 1.0
    torch.testing.assert_close(age, torch.tensor([0.0]))
    assert client.wait_calls == 1
    assert [request["current_frame_ids"] for request in client.submitted] == [
        [0],
        [16],
    ]


def test_central_semantic_generation_repair_is_bounded_when_packet_is_missing():
    class Client:
        def __init__(self):
            self.submitted = []
            self.published = 0
            self.wait_calls = 0

        def poll_latest(self, **kwargs):
            return None

        def publish(self, *args, **kwargs):
            self.published += 1

        def submit_latest(self, **kwargs):
            self.submitted.append(kwargs)

        def wait_latest(self, **kwargs):
            self.wait_calls += 1
            return None

    client = Client()
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter(
        "_device_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    )
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_non_blocking = True
    model._semantic_boundary_publish = False
    model._semantic_boundary_publish_interval = 1
    model._semantic_env_bootstrap_publish = True
    model._semantic_publish_interval_frames = 0
    model._semantic_last_episode_generations = {7: 0}
    model._semantic_last_published_frames = {7: (0, 16)}
    model._semantic_client = client
    model._semantic_cache = BatchFeature(
        data={"backbone_features": torch.full((1, 2, 3), 9.0)}
    )
    model._latest_semantic_metadata = {
        "env_ids": [7],
        "source_frame_ids": [16],
        "episode_generations": [0],
        "source_wallclock_s": [time.time()],
        "semantic_versions": [1],
    }
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([7]),
        "frame_ids": torch.tensor([0]),
        "episode_generations": torch.tensor([1]),
        "observation_wallclock_s": torch.tensor([time.time()]),
    }
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_fetch_hard_max_age_frames = 8
    model.output_action_chunks = 16
    model.compute_dtype = torch.float32

    outputs, _ = model._semantic_backbone(BatchFeature(data={}))

    assert outputs["backbone_features"].unique().item() == 9.0
    assert client.wait_calls == 6
    assert client.published == 3
    assert (
        sum(request.get("max_wait_ms") == 1000.0 for request in client.submitted) == 6
    )


def test_central_semantic_cache_reuses_completed_packet_without_waiting():
    class Client:
        def __init__(self):
            self.submitted = []
            self.wait_calls = 0
            self.fetch_calls = 0

        def poll_latest(self, **kwargs):
            return None

        def submit_latest(self, **kwargs):
            self.submitted.append(kwargs)

        def wait_latest(self, **kwargs):
            self.wait_calls += 1
            return (
                BatchFeature(data={"backbone_features": torch.full((1, 2, 3), 2.0)}),
                {
                    "env_ids": [7],
                    "source_frame_ids": [24],
                    "episode_generations": [0],
                    "source_wallclock_s": [time.time()],
                    "semantic_versions": [2],
                },
            )

        def fetch_latest(self, **kwargs):
            self.fetch_calls += 1
            return (
                BatchFeature(data={"backbone_features": torch.full((1, 2, 3), 3.0)}),
                {
                    "env_ids": [7],
                    "source_frame_ids": [40],
                    "episode_generations": [0],
                    "source_wallclock_s": [time.time()],
                    "semantic_versions": [3],
                },
            )

    client = Client()
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter(
        "_device_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    )
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_non_blocking = True
    model._semantic_boundary_publish = False
    model._semantic_boundary_publish_interval = 1
    model._semantic_env_bootstrap_publish = True
    model._semantic_publish_interval_frames = 0
    model._semantic_last_episode_generations = {7: 0}
    model._semantic_last_published_frames = {7: (0, 8)}
    model._semantic_client = client
    model._semantic_cache = BatchFeature(
        data={"backbone_features": torch.ones(1, 2, 3)}
    )
    model._latest_semantic_metadata = {
        "env_ids": [7],
        "source_frame_ids": [8],
        "episode_generations": [0],
        "source_wallclock_s": [time.time()],
        "semantic_versions": [1],
    }
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([7]),
        "frame_ids": torch.tensor([16]),
        "episode_generations": torch.tensor([0]),
        "observation_wallclock_s": torch.tensor([time.time()]),
    }
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_fetch_hard_max_age_frames = 8
    model.output_action_chunks = 16
    model.compute_dtype = torch.float32

    outputs, age = model._semantic_backbone(BatchFeature(data={}))

    assert outputs is model._semantic_cache
    torch.testing.assert_close(age, torch.tensor([0.4]))
    assert client.wait_calls == 0
    assert client.submitted[0]["current_frame_ids"] == [32]

    model._rollout_semantic_metadata["frame_ids"] = torch.tensor([32])
    outputs, age = model._semantic_backbone(BatchFeature(data={}))

    assert outputs["backbone_features"].unique().item() == 2.0
    torch.testing.assert_close(age, torch.tensor([0.4]))
    assert client.wait_calls == 1
    assert [request["current_frame_ids"] for request in client.submitted] == [
        [32],
        [32],
        [48],
    ]

    model._semantic_fetch_hard_max_age_frames = -1
    model._rollout_semantic_metadata["frame_ids"] = torch.tensor([48])
    outputs, age = model._semantic_backbone(BatchFeature(data={}))

    assert outputs["backbone_features"].unique().item() == 3.0
    torch.testing.assert_close(age, torch.tensor([0.4]))
    assert client.fetch_calls == 1
    assert client.wait_calls == 1


def test_central_semantic_cache_uses_latest_packet_after_bounded_age_wait():
    class Client:
        def __init__(self):
            self.submitted = []
            self.wait_calls = 0

        def poll_latest(self, **kwargs):
            return None

        def submit_latest(self, **kwargs):
            self.submitted.append(kwargs)

        def wait_latest(self, **kwargs):
            self.wait_calls += 1
            return (
                BatchFeature(data={"backbone_features": torch.ones(1, 2, 3)}),
                {
                    "env_ids": [7],
                    "source_frame_ids": [24],
                    "episode_generations": [0],
                    "source_wallclock_s": [time.time()],
                    "semantic_versions": [2],
                },
            )

    client = Client()
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter(
        "_device_anchor", torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    )
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_non_blocking = True
    model._semantic_boundary_publish = False
    model._semantic_boundary_publish_interval = 1
    model._semantic_env_bootstrap_publish = True
    model._semantic_publish_interval_frames = 0
    model._semantic_last_episode_generations = {7: 0}
    model._semantic_last_published_frames = {7: (0, 8)}
    model._semantic_client = client
    model._semantic_cache = BatchFeature(
        data={"backbone_features": torch.zeros(1, 2, 3)}
    )
    model._latest_semantic_metadata = {
        "env_ids": [7],
        "source_frame_ids": [8],
        "episode_generations": [0],
        "source_wallclock_s": [time.time()],
        "semantic_versions": [1],
    }
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([7]),
        "frame_ids": torch.tensor([32]),
        "episode_generations": torch.tensor([0]),
        "observation_wallclock_s": torch.tensor([time.time()]),
    }
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_fetch_hard_max_age_frames = 8
    model.output_action_chunks = 16
    model.compute_dtype = torch.float32

    outputs, age = model._semantic_backbone(BatchFeature(data={}))

    assert outputs["backbone_features"].unique().item() == 1.0
    torch.testing.assert_close(age, torch.tensor([0.4]))
    assert client.wait_calls == 1

    assert [request["current_frame_ids"] for request in client.submitted] == [
        [32],
        [48],
    ]


def test_dynamic_semantic_fetch_delay_tracks_observation_wallclock():
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    model._semantic_fetch_delay_fraction = 0.45
    model._semantic_fetch_delay_initial_s = 0.8
    model._semantic_fetch_delay_min_s = 0.1
    model._semantic_fetch_delay_max_s = 1.5
    model._semantic_fetch_delay_ema_alpha = 0.25
    model._semantic_observation_interval_ema_s = None
    model._semantic_last_observation_wallclock_s = None
    model._semantic_last_request_delay_s = 0.8

    first = model._update_semantic_fetch_request_delay([100.0, 100.0])
    second = model._update_semantic_fetch_request_delay([102.0, 102.0])
    third = model._update_semantic_fetch_request_delay([104.0, 104.0])
    ignored_pause = model._update_semantic_fetch_request_delay([112.0, 112.0])
    slower = model._update_semantic_fetch_request_delay([118.0, 118.0])
    clamped = model._update_semantic_fetch_request_delay([124.0, 124.0])

    assert first == pytest.approx(0.8)
    assert second == pytest.approx(0.9)
    assert third == pytest.approx(0.9)
    assert ignored_pause == pytest.approx(0.9)
    assert slower == pytest.approx(1.35)
    assert clamped == pytest.approx(1.5)


def test_env_bootstrap_skips_duplicate_control_publish_inputs():
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_boundary_publish = False
    model._semantic_env_bootstrap_publish = True
    model._semantic_publish_interval_frames = 0
    model._semantic_last_episode_generations = {}
    model._semantic_last_published_frames = {}
    model._semantic_cache = object()
    model._latest_semantic_metadata = {
        "env_ids": [1],
        "episode_generations": [0],
    }
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([1]),
        "frame_ids": torch.tensor([0]),
        "episode_generations": torch.tensor([0]),
    }

    assert not model._semantic_requires_publish_inputs()


def test_semantic_publish_interval_uses_simulator_frames():
    last_published = {1: (0, 0), 2: (0, 0)}

    assert not _semantic_publish_due(last_published, [1, 2], [0, 0], [4, 4], 8)
    assert _semantic_publish_due(last_published, [1, 2], [0, 0], [8, 8], 8)


def test_semantic_publish_interval_bootstraps_and_tracks_episode_generation():
    assert _semantic_publish_due({}, [1], [0], [0], 8)
    assert _semantic_publish_due(
        {
            1: (0, 12),
        },
        [1],
        [1],
        [0],
        8,
    )
    assert not _semantic_publish_due(
        {
            1: (1, 0),
        },
        [1],
        [1],
        [32],
        0,
    )


def test_action_only_observation_matches_action_head_shapes_without_images():
    state_config = SimpleNamespace(modality_keys=["joints"], exclude_state=False)
    action_config = SimpleNamespace(delta_indices=list(range(16)))

    class _StateProcessor:
        def apply_state(self, state, embodiment_tag):
            assert embodiment_tag == "libero_sim"
            return {"joints": state["joints"] * 2}

    processor = SimpleNamespace(
        modality_configs={
            "libero_sim": {"state": state_config, "action": action_config}
        },
        state_action_processor=_StateProcessor(),
        exclude_state=False,
        max_state_dim=29,
        max_action_horizon=50,
        embodiment_id_mapping={"libero_sim": 2},
    )
    result = _prepare_action_only_observation(
        processor,
        {"state.joints": np.ones((3, 7), dtype=np.float32)},
        SimpleNamespace(value="libero_sim"),
    )

    assert set(result) == {"state", "embodiment_id", "action_mask"}
    assert result["state"].shape == (3, 29)
    assert torch.all(result["state"][:, :7] == 2)
    assert torch.all(result["state"][:, 7:] == 0)
    assert result["embodiment_id"].tolist() == [2, 2, 2]
    assert torch.all(result["action_mask"][:, :16] == 1)
    assert torch.all(result["action_mask"][:, 16:] == 0)


def test_compute_evaluate_metrics_reports_per_task_success_across_ranks():
    metrics = compute_evaluate_metrics(
        [
            {
                "task_id": torch.tensor([0, 1]),
                "trial_id": torch.tensor([0, 0]),
                "success_once": torch.tensor([True, False]),
            },
            {
                "task_id": torch.tensor([0, 1, 1]),
                "trial_id": torch.tensor([1, 0, 2]),
                "success_once": torch.tensor([False, True, True]),
            },
        ],
        deduplicate_trials=True,
    )

    assert metrics["raw_num_trajectories"] == 5
    assert metrics["num_trajectories"] == 4
    assert metrics["success_once"] == pytest.approx(0.5)
    assert metrics["task/0/success_rate"] == pytest.approx(0.5)
    assert metrics["task/0/num_trajectories"] == 2
    assert metrics["task/1/success_rate"] == pytest.approx(0.5)
    assert metrics["task/1/num_trajectories"] == 2
    assert metrics["unique_task_trials"] == 4
    assert metrics["duplicate_task_trials"] == 1


def test_compute_evaluate_metrics_preserves_repeated_training_samples_by_default():
    metrics = compute_evaluate_metrics(
        [
            {
                "task_id": torch.tensor([1, 1]),
                "trial_id": torch.tensor([0, 0]),
                "success_once": torch.tensor([False, True]),
            }
        ]
    )

    assert metrics["raw_num_trajectories"] == 2
    assert metrics["num_trajectories"] == 2
    assert metrics["success_once"] == pytest.approx(0.5)
    assert metrics["task/1/num_trajectories"] == 2
    assert metrics["duplicate_task_trials"] == 1


def test_stale_age_gate_is_exactly_zero_through_threshold():
    ages_s = torch.tensor([0.0, 0.4, 0.8, 2.8])
    gate = _stale_age_gate(ages_s, control_hz=20.0, threshold_frames=8.0)

    assert gate.tolist() == pytest.approx([0.0, 0.0, 1.0, 6.0])


def test_stale_adapter_uses_semantic_context_only_above_age_threshold():
    class StateEncoder(torch.nn.Module):
        def forward(self, state, embodiment_id):
            del embodiment_id
            return torch.zeros(state.shape[0], 1, 6, dtype=state.dtype)

    head = object.__new__(FlowMatchingActionHeadForRLActionPrediction)
    torch.nn.Module.__init__(head)
    head.config = SimpleNamespace(state_history_length=1)
    head.model_action_dim = 2
    head.state_encoder = StateEncoder()
    head.packet_age_adapter = None
    head.packet_age_normalization_ms = 400.0
    head.action_history_length = 1
    head.action_history_adapter = None
    head.stale_residual_adapter = None
    head.stale_semantic_control_hz = 20.0
    head.stale_semantic_threshold_frames = 8.0
    head.stale_semantic_context_width = 4
    head.stale_semantic_adapter = torch.nn.Linear(7, 6, bias=False)
    with torch.no_grad():
        head.stale_semantic_adapter.weight.zero_()
        head.stale_semantic_adapter.weight[:, -4:] = 1.0

    action_input = BatchFeature(
        data={
            "state": torch.zeros(2, 1, 2),
            "packet_age_s": torch.tensor([0.4, 0.8]),
            "action_history": torch.zeros(2, 1, 2),
        }
    )
    semantic_features = torch.stack((torch.full((3, 4), 3.0), torch.ones(3, 4)))

    result = head._encode_state_features(
        action_input, embodiment_id=0, semantic_features=semantic_features
    )

    assert result[0].abs().sum().item() == pytest.approx(0.0)
    torch.testing.assert_close(result[1], torch.full_like(result[1], 4.0))


def test_stale_token_correction_is_identity_through_age_threshold():
    head = object.__new__(FlowMatchingActionHeadForRLActionPrediction)
    torch.nn.Module.__init__(head)
    head.model_action_dim = 2
    head.action_history_length = 1
    head.stale_semantic_control_hz = 20.0
    head.stale_semantic_threshold_frames = 8.0
    head.stale_semantic_context_width = 4
    head.stale_semantic_token_adapter = torch.nn.Linear(7, 4, bias=False)
    with torch.no_grad():
        head.stale_semantic_token_adapter.weight.zero_()
        head.stale_semantic_token_adapter.weight[:, :4] = torch.eye(4)

    semantic_features = torch.stack((torch.full((3, 4), 3.0), torch.full((3, 4), 2.0)))
    action_input = BatchFeature(
        data={
            "packet_age_s": torch.tensor([0.4, 0.8]),
            "action_history": torch.zeros(2, 1, 2),
        }
    )

    corrected = head._apply_stale_semantic_token_correction(
        semantic_features, action_input
    )

    torch.testing.assert_close(corrected[0], semantic_features[0])
    torch.testing.assert_close(corrected[1], torch.full_like(corrected[1], 4.0))


def test_fixed_age_eval_fetches_exact_simulator_frames(monkeypatch):
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter("_test_parameter", torch.nn.Parameter(torch.zeros(1)))
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_client = object.__new__(Gr00tN1d7SemanticCacheClient)
    model._semantic_eval_fixed_age_frames = 6
    model._semantic_eval_fixed_age_max_wait_ms = 1234.0
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_feature_tokens = 0
    model.output_action_chunks = 4
    model.compute_dtype = torch.float32
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([1, 2, 3, 4]),
        "frame_ids": torch.tensor([0, 4, 8, 12]),
        "episode_generations": torch.tensor([0, 1, 2, 3]),
    }
    captured = []

    def fetch_exact(**kwargs):
        captured.append(dict(kwargs))
        outputs = BatchFeature(data={"backbone_features": torch.ones(4, 2, 3)})
        return outputs, {"source_frame_ids": kwargs["source_frame_ids"]}

    monkeypatch.setattr(model._semantic_client, "fetch_exact", fetch_exact)
    fallback = BatchFeature(data={"backbone_features": torch.zeros(4, 2, 3)})

    outputs, age_s = model._fixed_age_eval_semantic(fallback, torch.full((4,), 99.0))

    assert captured[0]["source_frame_ids"] == [0, 0, 2, 6]
    assert captured[1]["source_frame_ids"] == [0, 2, 6, 10]
    assert captured[0]["episode_generations"] == [0, 1, 2, 3]
    assert captured[0]["max_wait_ms"] == 1234.0
    assert outputs["backbone_features"].unique().item() == 1.0
    torch.testing.assert_close(age_s, torch.tensor([0.0, 0.2, 0.3, 0.3]))


def test_exact_age_train_uses_env_scheduled_packets(monkeypatch):
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter("_test_parameter", torch.nn.Parameter(torch.zeros(1)))
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_client = object.__new__(Gr00tN1d7SemanticCacheClient)
    model._semantic_train_random_age_min_frames = 0
    model._semantic_train_random_age_max_frames = 6
    model._semantic_train_fixed_age_max_wait_ms = 4321.0
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_feature_tokens = 0
    model.compute_dtype = torch.float32
    model._latest_semantic_metadata = {}
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([1, 2, 3]),
        "frame_ids": torch.tensor([8, 8, 8]),
        "episode_generations": torch.tensor([0, 1, 2]),
        "target_age_frames": torch.tensor([0, 3, 6]),
    }
    captured = []

    def fetch_exact(**kwargs):
        captured.append(dict(kwargs))
        outputs = BatchFeature(data={"backbone_features": torch.ones(3, 2, 3)})
        return outputs, {
            "source_frame_ids": kwargs["source_frame_ids"],
            "semantic_versions": [11, 12, 13],
        }

    monkeypatch.setattr(model._semantic_client, "fetch_exact", fetch_exact)
    fallback = BatchFeature(data={"backbone_features": torch.zeros(3, 2, 3)})

    outputs, age_s = model._fixed_age_train_semantic(fallback, torch.full((3,), 99.0))

    assert captured[0]["source_frame_ids"] == [8, 5, 2]
    assert captured[0]["max_wait_ms"] == 4321.0
    assert outputs["backbone_features"].unique().item() == 1.0
    assert model._latest_semantic_metadata["semantic_versions"] == [11, 12, 13]
    torch.testing.assert_close(age_s, torch.tensor([0.0, 0.15, 0.3]))


def test_fixed_age_eval_fails_closed_when_exact_packet_is_missing(monkeypatch):
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter("_test_parameter", torch.nn.Parameter(torch.zeros(1)))
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_client = object.__new__(Gr00tN1d7SemanticCacheClient)
    model._semantic_eval_fixed_age_frames = 6
    model._semantic_eval_fixed_age_max_wait_ms = 10.0
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model.compute_dtype = torch.float32
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([1]),
        "frame_ids": torch.tensor([8]),
        "episode_generations": torch.tensor([0]),
    }
    monkeypatch.setattr(model._semantic_client, "fetch_exact", lambda **_: None)
    fallback = BatchFeature(data={"backbone_features": torch.zeros(1, 2, 3)})

    with pytest.raises(RuntimeError, match="could not fetch exact packets"):
        model._fixed_age_eval_semantic(fallback, torch.zeros(1))


def test_random_age_eval_is_bounded_and_reproducible(monkeypatch):
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    torch.nn.Module.__init__(model)
    model.register_parameter("_test_parameter", torch.nn.Parameter(torch.zeros(1)))
    model._semantic_enabled = True
    model._semantic_central_cache = True
    model._semantic_client = object.__new__(Gr00tN1d7SemanticCacheClient)
    model._semantic_eval_fixed_age_frames = -1
    model._semantic_eval_random_age_min_frames = 0
    model._semantic_eval_random_age_max_frames = 6
    model._semantic_eval_random_age_seed = 2026
    model._semantic_eval_fixed_age_max_wait_ms = 1234.0
    model._semantic_age_mode = "simulator"
    model._semantic_control_hz = 20.0
    model._semantic_feature_tokens = 0
    model.output_action_chunks = 16
    model.compute_dtype = torch.float32
    model._rollout_semantic_metadata = {
        "env_ids": torch.tensor([1]),
        "frame_ids": torch.tensor([16]),
        "episode_generations": torch.tensor([0]),
    }
    captured = []

    def fetch_exact(**kwargs):
        captured.append(dict(kwargs))
        outputs = BatchFeature(data={"backbone_features": torch.ones(1, 2, 3)})
        return outputs, {"source_frame_ids": kwargs["source_frame_ids"]}

    monkeypatch.setattr(model._semantic_client, "fetch_exact", fetch_exact)
    fallback = BatchFeature(data={"backbone_features": torch.zeros(1, 2, 3)})

    _, first_age_s = model._fixed_age_eval_semantic(fallback, torch.full((1,), 99.0))
    first_source = captured[0]["source_frame_ids"]
    captured.clear()
    _, second_age_s = model._fixed_age_eval_semantic(fallback, torch.full((1,), 99.0))

    assert first_source == captured[0]["source_frame_ids"]
    assert 0 <= first_age_s.item() * 20 <= 6
    torch.testing.assert_close(first_age_s, second_age_s)


def test_action_history_keeps_latest_executed_frames_from_chunk():
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    model._action_history = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    model._semantic_central_cache = False
    model._current_action_history_keys = []

    model._append_action_history(torch.tensor([[[10.0], [11.0], [12.0]]]))

    assert model._action_history.squeeze(-1).tolist() == [[4.0, 10.0, 11.0, 12.0]]


def test_action_history_ignores_unexecuted_prediction_tail():
    model = object.__new__(GR00T_N1_7_ForRLActionPrediction)
    model.output_action_chunks = 2
    model._action_history = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
    model._semantic_central_cache = False
    model._current_action_history_keys = []

    model._append_executed_action_history(
        torch.tensor([[[10.0], [11.0], [12.0], [13.0], [14.0]]])
    )

    assert model._action_history.squeeze(-1).tolist() == [[3.0, 4.0, 10.0, 11.0]]


def test_semantic_proxy_merges_raw_observation_waves():
    requests = []
    for offset in (0, 2):
        requests.append(
            {
                "data": {
                    "observation": {
                        "states": torch.arange(offset, offset + 2).reshape(2, 1),
                        "main_images": torch.full((2, 1, 1, 1), offset),
                        "wrist_images": torch.full((2, 1, 1, 1), offset + 1),
                        "task_descriptions": [f"task-{offset}", f"task-{offset + 1}"],
                    },
                    "metadata": {
                        "env_ids": [offset, offset + 1],
                        "frame_ids": [8, 8],
                        "episode_generations": [0, 0],
                        "observation_wallclock_s": [1.0, 1.0],
                        "semantic_priority": offset,
                    },
                }
            }
        )

    observation, metadata = Gr00tN1d7SemanticPreprocessProxy._merge_raw_requests(
        requests
    )

    assert observation["states"].flatten().tolist() == [0, 1, 2, 3]
    assert observation["task_descriptions"] == ["task-0", "task-1", "task-2", "task-3"]
    assert metadata["env_ids"] == [0, 1, 2, 3]
    assert metadata["semantic_priority"] == 0


def test_semantic_proxy_batch_target_releases_pending_wave():
    proxy = object.__new__(Gr00tN1d7SemanticPreprocessProxy)
    proxy.running = True
    proxy.batch_max_requests = 2
    proxy.batch_target_envs = 4
    proxy.batch_wait_ms = 1000.0
    proxy._pending_lock = threading.Lock()
    proxy._pending_condition = threading.Condition(proxy._pending_lock)
    submitted = time.perf_counter()
    proxy._pending_raw = {
        (0, 1): {"env_count": 2, "submitted_perf": submitted},
        (2, 3): {"env_count": 2, "submitted_perf": submitted},
    }

    selected = proxy._take_raw_batch()

    assert len(selected) == 2
    assert proxy._pending_raw == {}


def test_semantic_proxy_text_padding_uses_libero_compact_length():
    inputs = {
        "input_ids": torch.ones(2, 156, dtype=torch.int64),
        "attention_mask": torch.ones(2, 156, dtype=torch.int64),
    }

    padded = Gr00tN1d7SemanticPreprocessProxy._canonicalize_text_inputs(inputs, 160)

    assert padded["input_ids"].shape == (2, 160)
    assert padded["attention_mask"].shape == (2, 160)
    assert padded["attention_mask"][:, :156].all()
    assert not padded["attention_mask"][:, 156:].any()


def test_semantic_server_caps_merged_forward_by_environment_count():
    packets = [
        ((0,), {"env_ids": list(range(0, 60))}),
        ((1,), {"env_ids": list(range(60, 120))}),
        ((2,), {"env_ids": list(range(120, 160))}),
    ]

    selected = Gr00tN1d7SemanticBackbonePolicy._select_pending_packets(
        packets,
        max_requests=12,
        max_envs=60,
    )

    assert selected == packets[:1]
    assert sum(len(packet["env_ids"]) for _, packet in selected) == 60


def test_semantic_server_prunes_batch_once_every_env_has_a_newer_pending_frame():
    policy = object.__new__(Gr00tN1d7SemanticBackbonePolicy)
    policy._cache_lock = threading.RLock()
    policy.pending_batches = {}
    policy.semantic_cache_by_env = {}

    first = BatchFeature(
        data={"input_ids": torch.tensor([[10], [20]], dtype=torch.int64)}
    )
    second = BatchFeature(
        data={"input_ids": torch.tensor([[200], [300]], dtype=torch.int64)}
    )
    third = BatchFeature(data={"input_ids": torch.tensor([[100]], dtype=torch.int64)})

    assert (
        policy._queue_observations(
            first,
            {
                "env_ids": [1, 2],
                "frame_ids": [0, 0],
                "episode_generations": [0, 0],
            },
        )["accepted"]
        == 2
    )
    assert (
        policy._queue_observations(
            second,
            {
                "env_ids": [2, 3],
                "frame_ids": [16, 16],
                "episode_generations": [0, 0],
            },
        )["accepted"]
        == 2
    )
    assert policy.pending_batches[(1, 2)]["env_ids"] == [1]
    assert policy.pending_batches[(1, 2)]["source_frame_ids"] == [0]
    assert policy.pending_batches[(1, 2)]["inputs"]["input_ids"].tolist() == [[10]]
    assert policy.pending_batches[(2, 3)]["env_ids"] == [2, 3]
    assert (
        policy._queue_observations(
            third,
            {
                "env_ids": [1],
                "frame_ids": [16],
                "episode_generations": [0],
            },
        )["accepted"]
        == 1
    )

    assert set(policy.pending_batches) == {(2, 3), (1,)}
    assert policy.pending_batches[(2, 3)]["source_frame_ids"] == [16, 16]
    assert policy.pending_batches[(1,)]["source_frame_ids"] == [16]


def test_prefetched_eval_bootstrap_is_consumed_without_second_reset():
    worker = object.__new__(EnvWorker)
    worker._prefetched_eval_bootstrap = [{"obs": {"cached": True}}]
    worker._reset_eval_bootstrap = lambda stage_id: pytest.fail(
        f"unexpected reset for stage {stage_id}"
    )

    bootstrap = worker._take_eval_bootstrap(0)

    assert bootstrap == {"obs": {"cached": True}}
    assert worker._prefetched_eval_bootstrap == [None]


def test_continued_eval_bootstrap_uses_prepared_reset_state_ids():
    worker = object.__new__(EnvWorker)
    env = SimpleNamespace(
        reset_state_ids=np.asarray([12, 34], dtype=np.int64),
        reset=MagicMock(return_value=({"obs": torch.ones(2, 1)}, {})),
    )
    worker.eval_env_list = [env]
    worker.eval_prev_done = [torch.ones(2, dtype=torch.bool)]
    worker.eval_num_envs_per_stage = 2
    worker._build_rollout_input_data = lambda data, stage_id, mode: {
        "data": data,
        "stage_id": stage_id,
        "mode": mode,
    }

    bootstrap = worker._continue_eval_bootstrap(0)

    np.testing.assert_array_equal(
        env.reset.call_args.kwargs["reset_state_ids"], np.asarray([12, 34])
    )
    assert not worker.eval_prev_done[0].any()
    assert bootstrap["stage_id"] == 0
    assert bootstrap["mode"] == "eval"


def test_non_auto_reset_eval_restarts_once_then_continues_cursor():
    worker = object.__new__(EnvWorker)
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker._timer_metrics = {}
    worker.eval_rollout_epoch = 3
    worker.stage_num = 1
    worker.n_eval_chunk_steps = 0
    worker.env_decoupled_mode = False
    worker.eval_enable_offload = False
    worker.cfg = SimpleNamespace(
        env=SimpleNamespace(eval=SimpleNamespace(auto_reset=False)),
        rollout=SimpleNamespace(group_name="rollout"),
    )
    worker._take_eval_bootstrap = MagicMock(return_value={"epoch": 0})
    worker._continue_eval_bootstrap = MagicMock(
        side_effect=({"epoch": 1}, {"epoch": 2})
    )
    worker.send_to = MagicMock()
    worker.finish_rollout = MagicMock()
    worker.eval_env_list = [SimpleNamespace()]

    metrics = worker.evaluate(input_channel=None, rollout_channel=None)

    assert metrics == {}
    worker._take_eval_bootstrap.assert_called_once_with(0)
    assert worker._continue_eval_bootstrap.call_count == 2
    assert [call.kwargs["data"]["epoch"] for call in worker.send_to.call_args_list] == [
        0,
        1,
        2,
    ]


def test_semantic_mid_chunk_frames_stagger_across_env_ranks():
    frames = [
        _staggered_semantic_frame(8, rank, world_size=6, enabled=True)
        for rank in range(6)
    ]

    assert frames == [1, 2, 3, 5, 6, 8]
    assert _staggered_semantic_frame(8, rank=3, world_size=6, enabled=False) == 8
    assert _staggered_semantic_frame(8, rank=0, world_size=1, enabled=True) == 8
    assert [
        _staggered_semantic_frame(13, rank, world_size=6, enabled=True, min_frame=8)
        for rank in range(6)
    ] == [8, 9, 10, 11, 12, 13]


def test_eval_semantic_publish_frame_override_is_independent_from_train_stagger():
    train_frame = _staggered_semantic_frame(
        16, rank=2, world_size=3, enabled=True, min_frame=10
    )

    assert train_frame == 16
    assert _resolve_semantic_eval_publish_frame(train_frame, -1) == 16
    assert _resolve_semantic_eval_publish_frame(train_frame, 10) == 10


def test_semantic_mid_chunk_frame_must_be_reachable():
    _validate_semantic_publish_frame(True, publish_frame=2, execution_horizon=4)

    with pytest.raises(ValueError, match="must be reachable"):
        _validate_semantic_publish_frame(True, publish_frame=10, execution_horizon=4)

    _validate_semantic_publish_frame(False, publish_frame=10, execution_horizon=4)


def test_eval_bootstrap_falls_back_to_reset_after_cache_is_consumed():
    worker = object.__new__(EnvWorker)
    worker._prefetched_eval_bootstrap = [None]
    worker._reset_eval_bootstrap = lambda stage_id: {"stage_id": stage_id}

    assert worker._take_eval_bootstrap(0) == {"stage_id": 0}


def test_env_boundary_publish_uses_latest_frame_without_advancing_clock():
    worker = object.__new__(EnvWorker)
    worker._rank = 0
    worker.stage_num = 1
    worker.train_num_envs_per_stage = 2
    worker.eval_num_envs_per_stage = 2
    worker._semantic_env_clock = {
        ("train", 0): {
            "frame_ids": torch.tensor([0, 16]),
            "generations": torch.tensor([0, 3]),
        }
    }
    captured = {}
    worker._publish_semantic_observation = lambda obs, metadata: captured.update(
        metadata
    )

    worker._publish_boundary_semantic(
        {"elapsed_steps": torch.tensor([16, 0])}, stage_id=0, mode="train"
    )

    torch.testing.assert_close(captured["frame_ids"], torch.tensor([16, 0]))
    torch.testing.assert_close(captured["episode_generations"], torch.tensor([0, 4]))
    torch.testing.assert_close(
        worker._semantic_env_clock[("train", 0)]["frame_ids"],
        torch.tensor([0, 16]),
    )
    assert captured["semantic_priority"] == 1


def test_env_semantic_publisher_forwards_priority():
    published = {}
    publisher = SimpleNamespace(
        poll=lambda: None,
        publish=lambda observation, metadata: published.update(
            observation=observation, metadata=metadata
        ),
    )
    worker = object.__new__(EnvWorker)
    worker._semantic_raw_publisher = publisher
    obs = {
        "states": torch.zeros(2, 1),
        "main_images": torch.zeros(2, 1),
        "wrist_images": torch.zeros(2, 1),
        "task_descriptions": np.array(["a", "b"]),
    }
    metadata = {
        "env_ids": torch.tensor([1, 2]),
        "frame_ids": torch.tensor([16, 16]),
        "episode_generations": torch.tensor([0, 0]),
        "observation_wallclock_s": torch.tensor([1.0, 1.0]),
        "semantic_priority": 1,
    }

    worker._publish_semantic_observation(obs, metadata)

    assert published["metadata"]["semantic_priority"] == 1
    assert published["metadata"]["frame_ids"] == [16, 16]
