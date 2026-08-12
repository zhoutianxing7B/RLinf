# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import pickle
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zmq
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from transformers import AutoConfig, AutoModel
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.simulation_io import OBS_CONVERSION

_TORCH_DTYPES = {
    "bool": torch.bool,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}


def _tensor_to_payload(tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().contiguous().cpu()
    dtype_name = str(tensor.dtype).replace("torch.", "")
    if tensor.dtype == torch.bfloat16:
        array = tensor.view(torch.uint16).numpy()
    else:
        array = tensor.numpy()
    return {
        "__torch_tensor__": True,
        "dtype": dtype_name,
        "shape": list(tensor.shape),
        "data": array,
    }


def _payload_to_tensor(
    payload: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    floating_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not isinstance(payload, dict) or not payload.get("__torch_tensor__"):
        raise TypeError(f"Expected encoded tensor payload, got {type(payload)!r}")
    dtype_name = str(payload["dtype"])
    shape = tuple(int(dim) for dim in payload["shape"])
    array = payload["data"]
    if dtype_name == "bfloat16":
        tensor = torch.from_numpy(np.asarray(array, dtype=np.uint16)).view(
            torch.bfloat16
        )
    else:
        dtype = _TORCH_DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError(
                f"Unsupported tensor dtype from semantic server: {dtype_name}"
            )
        tensor = torch.from_numpy(np.asarray(array)).to(dtype=dtype)
    tensor = tensor.reshape(shape)
    if floating_dtype is not None and torch.is_floating_point(tensor):
        tensor = tensor.to(dtype=floating_dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor


def batch_feature_to_payload(
    batch: BatchFeature | dict[str, torch.Tensor],
) -> dict[str, Any]:
    return {key: _tensor_to_payload(value) for key, value in dict(batch).items()}


def payload_to_batch_feature(
    payload: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    floating_dtype: torch.dtype | None = None,
) -> BatchFeature:
    return BatchFeature(
        data={
            key: _payload_to_tensor(value, device=device, floating_dtype=floating_dtype)
            for key, value in payload.items()
        }
    )


_TRANSPORT_SCALE_PREFIX = "__rlinf_transport_scale__"


def quantize_semantic_transport(
    batch: BatchFeature | dict[str, torch.Tensor], mode: str
) -> BatchFeature:
    """Compress large semantic tokens for RPC without changing the model interface."""
    mode = str(mode).lower()
    data = dict(batch)
    if mode == "none":
        return BatchFeature(data=data)
    if mode != "int8":
        raise ValueError(f"Unsupported semantic transport quantization: {mode}")
    features = data.get("backbone_features")
    if features is None or not torch.is_floating_point(features):
        raise ValueError("INT8 semantic transport requires floating backbone_features")
    features_fp32 = features.float()
    scale = features_fp32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 127.0
    data["backbone_features"] = (
        (features_fp32 / scale).round().clamp(-127, 127).to(torch.int8)
    )
    data[f"{_TRANSPORT_SCALE_PREFIX}backbone_features"] = scale.to(torch.bfloat16)
    return BatchFeature(data=data)


def dequantize_semantic_transport(
    batch: BatchFeature | dict[str, torch.Tensor], dtype: torch.dtype
) -> BatchFeature:
    data = dict(batch)
    scale_key = f"{_TRANSPORT_SCALE_PREFIX}backbone_features"
    scale = data.pop(scale_key, None)
    if scale is not None:
        data["backbone_features"] = data["backbone_features"].to(dtype) * scale.to(
            dtype
        )
    return BatchFeature(data=data)


def _snapshot_nested_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, dict):
        return {key: _snapshot_nested_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_nested_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_nested_cpu(item) for item in value)
    return value


def _split_csv_values(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = str(value).split(",")
    return [str(item).strip() for item in raw_items if str(item).strip()]


def _parse_semantic_endpoints(host: Any, port: Any) -> list[tuple[str, int]]:
    hosts = _split_csv_values(host)
    ports = _split_csv_values(port)
    if not hosts:
        hosts = ["127.0.0.1"]
    if not ports:
        ports = ["6666"]
    if len(hosts) == 1 and len(ports) > 1:
        hosts = hosts * len(ports)
    elif len(ports) == 1 and len(hosts) > 1:
        ports = ports * len(hosts)
    elif len(hosts) != len(ports):
        raise ValueError(
            f"semantic server hosts/ports must have matching lengths or one side must be scalar: "
            f"hosts={hosts!r} ports={ports!r}"
        )
    return [(host, int(port)) for host, port in zip(hosts, ports, strict=True)]


class _ZmqRpcClient:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6666,
        timeout_ms: int = 60000,
        api_token: str | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self._closed = False
        self.context = zmq.Context()
        self._init_socket()

    def _init_socket(self) -> None:
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call_endpoint(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request = {"endpoint": endpoint, "data": data or {}}
        if self.api_token is not None:
            request["api_token"] = self.api_token
        try:
            self.socket.send(pickle.dumps(request, protocol=pickle.HIGHEST_PROTOCOL))
            response = pickle.loads(self.socket.recv())
        except zmq.error.Again:
            self.socket.close(linger=0)
            self._init_socket()
            raise
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Semantic server error: {response['error']}")
        return response

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.socket.close(linger=0)
        self.context.term()


class _ZmqRpcServer:
    def __init__(
        self,
        policy: Any,
        host: str = "*",
        port: int = 6666,
        publish_port: int | None = None,
        api_token: str | None = None,
        batch_max_requests: int = 8,
        batch_target_requests: int = 0,
        batch_target_envs: int = 0,
        batch_wait_ms: float = 2.0,
        bootstrap_target_envs: int = 0,
        bootstrap_wait_ms: float = 30000.0,
        rpc_batch_wait_ms: float = 2.0,
    ):
        self.policy = policy
        self.host = host
        self.port = int(port)
        self.publish_port = int(publish_port if publish_port is not None else port + 1)
        self.api_token = api_token
        self.batch_max_requests = max(1, int(batch_max_requests))
        self.batch_target_requests = min(
            self.batch_max_requests, max(0, int(batch_target_requests))
        )
        self.batch_target_envs = max(0, int(batch_target_envs))
        self.batch_wait_ms = max(0.0, float(batch_wait_ms))
        self.bootstrap_target_envs = max(0, int(bootstrap_target_envs))
        self.bootstrap_wait_ms = max(0.0, float(bootstrap_wait_ms))
        self._bootstrap_complete = self.bootstrap_target_envs <= 0
        self._bootstrap_deadline_perf: float | None = None
        self.rpc_batch_wait_ms = max(0.0, float(rpc_batch_wait_ms))
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://{host}:{port}")
        self._scheduler_wakeup = threading.Event()
        self.policy.scheduler_wakeup_callback = self._scheduler_wakeup.set
        self._scheduler_thread: threading.Thread | None = None
        self._ingest_ready = threading.Event()
        self._ingest_thread: threading.Thread | None = None
        self._fetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.batch_max_requests,
            thread_name_prefix="gr00t-semantic-fetch",
        )
        self._fetch_futures: dict[concurrent.futures.Future[Any], list[bytes]] = {}

    def _validate_token(self, request: dict[str, Any]) -> bool:
        return self.api_token is None or request.get("api_token") == self.api_token

    def _send(self, envelope: list[bytes], result: Any) -> None:
        self.socket.send_multipart(
            [*envelope, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)]
        )

    @staticmethod
    def _batch_signature(request: dict[str, Any]) -> tuple:
        payload = request.get("data", {}).get("backbone_inputs", {})
        return tuple(
            (key, tuple(value.get("shape", [])[1:]), value.get("dtype"))
            for key, value in sorted(payload.items())
        )

    def _recv_request(self) -> tuple[list[bytes], dict[str, Any]]:
        frames = self.socket.recv_multipart()
        return frames[:-1], pickle.loads(frames[-1])

    def _submit_fetch(
        self, envelope: list[bytes], endpoint: str, data: dict[str, Any]
    ) -> None:
        fetch = (
            self.policy.fetch_latest
            if endpoint == "fetch_latest"
            else self.policy.fetch_exact
        )
        future = self._fetch_executor.submit(fetch, data)
        self._fetch_futures[future] = envelope

    def _drain_fetch_responses(self) -> None:
        completed = [future for future in self._fetch_futures if future.done()]
        for future in completed:
            envelope = self._fetch_futures.pop(future)
            try:
                result = future.result()
            except Exception as exc:
                logging.exception("Semantic fetch request failed")
                result = {"error": str(exc)}
            self._send(envelope, result)

    def _run_scheduler(self) -> None:
        if self.policy.device.type == "cuda":
            torch.cuda.set_device(self.policy.device)
        while self.running:
            self._scheduler_wakeup.wait(timeout=0.01)
            self._scheduler_wakeup.clear()
            while self.running:
                ready = self._wait_for_ready_batch()
                if ready <= 0:
                    break
                try:
                    processed = self.policy.process_pending(
                        self.batch_max_requests,
                        max_batch_envs=self.batch_target_envs,
                    )
                except Exception as exc:
                    self.policy.last_scheduler_error = str(exc)
                    logging.exception("Semantic scheduler forward failed")
                    break
                if processed <= 0:
                    break

    def _wait_for_ready_batch(self) -> int:
        deadline = time.perf_counter() + self.batch_wait_ms / 1000.0
        while self.running:
            ready, total = self.policy.pending_request_counts()
            ready_envs, total_envs = self.policy.pending_env_counts()
            if total <= 0:
                return 0
            if not self._bootstrap_complete:
                if self._bootstrap_deadline_perf is None:
                    self._bootstrap_deadline_perf = (
                        time.perf_counter() + self.bootstrap_wait_ms / 1000.0
                    )
                if ready_envs >= self.bootstrap_target_envs:
                    self._bootstrap_complete = True
                    return ready
                if time.perf_counter() >= self._bootstrap_deadline_perf and ready > 0:
                    logging.warning(
                        "Semantic bootstrap timed out with %d/%d ready envs "
                        "(%d total); falling back to steady-state batching",
                        ready_envs,
                        self.bootstrap_target_envs,
                        total_envs,
                    )
                    self._bootstrap_complete = True
                    return ready
                self._scheduler_wakeup.wait(timeout=0.002)
                self._scheduler_wakeup.clear()
                continue
            # A freshness request raises packet priority, but it must not bypass
            # the short coalescing window. Immediate dispatch fragments requests
            # arriving from different rollout ranks into many small VLM forwards.
            self.policy.freshness_demand_active()
            if self.batch_target_envs > 0 and ready_envs >= self.batch_target_envs:
                return ready
            if (
                self.batch_target_envs <= 0
                and self.batch_target_requests > 0
                and ready >= self.batch_target_requests
            ):
                return ready
            if self.batch_target_envs <= 0 and self.batch_target_requests <= 0:
                if self.batch_wait_ms <= 0 or time.perf_counter() >= deadline:
                    return ready
            if time.perf_counter() >= deadline and ready > 0:
                return ready
            self._scheduler_wakeup.wait(timeout=0.002)
            self._scheduler_wakeup.clear()
        return 0

    def _run_ingest(self) -> None:
        socket = self.context.socket(zmq.ROUTER)
        socket.bind(f"tcp://{self.host}:{self.publish_port}")
        self._ingest_ready.set()
        try:
            while self.running:
                if not socket.poll(100, zmq.POLLIN):
                    continue
                frames = socket.recv_multipart()
                envelope, request = frames[:-1], pickle.loads(frames[-1])
                if not self._validate_token(request):
                    result = {"error": "unauthorized"}
                elif request.get("endpoint") == "publish_observations":
                    result = self.policy.publish_observations(request.get("data", {}))
                    self._scheduler_wakeup.set()
                elif request.get("endpoint") == "publish_raw_observations":
                    result = self.policy.publish_raw_observations(
                        request.get("data", {})
                    )
                    self._scheduler_wakeup.set()
                else:
                    endpoint = request.get("endpoint")
                    result = {"error": f"Unknown ingest endpoint: {endpoint}"}
                socket.send_multipart(
                    [*envelope, pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)]
                )
        except Exception:
            if self.running:
                logging.exception("Semantic ingest request failed")
        finally:
            socket.close(linger=0)

    def run(self) -> None:
        self._ingest_thread = threading.Thread(
            target=self._run_ingest,
            name="gr00t-semantic-ingest",
            daemon=True,
        )
        self._ingest_thread.start()
        if not self._ingest_ready.wait(timeout=10.0):
            raise RuntimeError("Semantic ingest server failed to start")
        logging.info(
            "Semantic server listening on %s, publish tcp://%s:%d "
            "(batch_max=%d, batch_target_requests=%d, batch_target_envs=%d, "
            "wait_ms=%.2f, bootstrap_target_envs=%d, bootstrap_wait_ms=%.2f)",
            self.socket.getsockopt_string(zmq.LAST_ENDPOINT),
            self.host,
            self.publish_port,
            self.batch_max_requests,
            self.batch_target_requests,
            self.batch_target_envs,
            self.batch_wait_ms,
            self.bootstrap_target_envs,
            self.bootstrap_wait_ms,
        )
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            name="gr00t-semantic-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()
        while self.running:
            pending: list[tuple[list[bytes], dict[str, Any]]] = []
            try:
                self._drain_fetch_responses()
                if not self.socket.poll(2, zmq.POLLIN):
                    continue
                pending.append(self._recv_request())
                deadline = time.perf_counter() + self.rpc_batch_wait_ms / 1000.0
                while len(pending) < self.batch_max_requests:
                    remaining_ms = max(0, int((deadline - time.perf_counter()) * 1000))
                    if not self.socket.poll(remaining_ms, zmq.POLLIN):
                        break
                    pending.append(self._recv_request())

                encode_groups: dict[
                    tuple, list[tuple[list[bytes], dict[str, Any]]]
                ] = {}
                for envelope, request in pending:
                    if not self._validate_token(request):
                        self._send(envelope, {"error": "unauthorized"})
                        continue
                    endpoint = request.get("endpoint")
                    if endpoint == "ping":
                        self._send(envelope, {"status": "ok"})
                    elif endpoint == "kill":
                        self.running = False
                        self._send(envelope, {"status": "stopping"})
                    elif endpoint == "encode_backbone":
                        signature = self._batch_signature(request)
                        encode_groups.setdefault(signature, []).append(
                            (envelope, request)
                        )
                    elif endpoint == "publish_observations":
                        result = self.policy.publish_observations(
                            request.get("data", {})
                        )
                        self._scheduler_wakeup.set()
                        self._send(envelope, result)
                    elif endpoint == "publish_raw_observations":
                        result = self.policy.publish_raw_observations(
                            request.get("data", {})
                        )
                        self._scheduler_wakeup.set()
                        self._send(envelope, result)
                    elif endpoint == "fetch_latest":
                        self._submit_fetch(envelope, endpoint, request.get("data", {}))
                    elif endpoint == "fetch_exact":
                        self._submit_fetch(envelope, endpoint, request.get("data", {}))
                    else:
                        self._send(envelope, {"error": f"Unknown endpoint: {endpoint}"})

                for group in encode_groups.values():
                    try:
                        results = self.policy.encode_backbone_batch(
                            [request.get("data", {}) for _, request in group]
                        )
                        for (envelope, _), result in zip(group, results, strict=True):
                            self._send(envelope, result)
                    except Exception as exc:
                        logging.exception("Semantic server batch failed")
                        for envelope, _ in group:
                            self._send(envelope, {"error": str(exc)})
                self._drain_fetch_responses()
            except Exception as exc:
                logging.exception("Semantic server request failed")
                for envelope, _ in pending:
                    self._send(envelope, {"error": str(exc)})

    def close(self) -> None:
        self.running = False
        self._scheduler_wakeup.set()
        self._fetch_executor.shutdown(wait=False, cancel_futures=True)
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=5.0)
        if self._ingest_thread is not None:
            self._ingest_thread.join(timeout=5.0)
        self.socket.close(linger=0)
        self.context.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class Gr00tN1d7SemanticBackboneClient:
    """Client for one or more GR00T N1.7 VLM/backbone-only servers."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | str = 6666,
        timeout_ms: int = 60000,
        api_token: str | None = None,
    ):
        self.endpoints = _parse_semantic_endpoints(host, port)
        self.clients = [
            _ZmqRpcClient(
                host=endpoint_host,
                port=endpoint_port,
                timeout_ms=timeout_ms,
                api_token=api_token,
            )
            for endpoint_host, endpoint_port in self.endpoints
        ]
        self._next_client_index = os.getpid() % max(len(self.clients), 1)
        self.client_id = f"{os.uname().nodename}:{os.getpid()}:{id(self)}"
        self.last_metrics: dict[str, float] = {}
        self.last_response_metadata: dict[str, Any] = {}

    def _pick_client(self) -> tuple[int, _ZmqRpcClient]:
        index = self._next_client_index
        self._next_client_index = (self._next_client_index + 1) % len(self.clients)
        return index, self.clients[index]

    def encode_backbone(
        self,
        backbone_inputs: BatchFeature,
        *,
        metadata: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        floating_dtype: torch.dtype | None = None,
    ) -> BatchFeature:
        started = time.perf_counter()
        client_index, client = self._pick_client()
        response = client.call_endpoint(
            "encode_backbone",
            {
                "backbone_inputs": batch_feature_to_payload(backbone_inputs),
                "metadata": {"client_id": self.client_id, **dict(metadata or {})},
            },
        )
        outputs = payload_to_batch_feature(
            response["backbone_outputs"],
            device=device,
            floating_dtype=floating_dtype,
        )
        outputs = dequantize_semantic_transport(
            outputs, floating_dtype or torch.bfloat16
        )
        endpoint_host, endpoint_port = self.endpoints[client_index]
        metrics = dict(response.get("metrics", {}))
        metrics["semantic_server_roundtrip_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        metrics["semantic_server_endpoint_index"] = float(client_index)
        metrics["semantic_server_endpoint_port"] = float(endpoint_port)
        self.last_metrics = metrics
        self.last_response_metadata = dict(response.get("metadata", {}))
        return outputs

    def close(self) -> None:
        for client in self.clients:
            client.close()


class Gr00tN1d7SemanticCacheClient:
    """Publish and fetch a server-owned semantic cache sharded by environment."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | str = 6666,
        publish_port: int | str | None = None,
        timeout_ms: int = 60000,
        api_token: str | None = None,
        fetch_target_age_frames: int = 0,
        fetch_max_wait_ms: float = 0.0,
    ):
        self.fetch_target_age_frames = max(0, int(fetch_target_age_frames))
        self.fetch_max_wait_ms = max(0.0, float(fetch_max_wait_ms))
        self.endpoints = _parse_semantic_endpoints(host, port)
        if publish_port is None:
            self.publish_endpoints = [
                (endpoint_host, endpoint_port + 1)
                for endpoint_host, endpoint_port in self.endpoints
            ]
        else:
            self.publish_endpoints = _parse_semantic_endpoints(host, publish_port)
        if len(self.publish_endpoints) != len(self.endpoints):
            raise ValueError(
                "Semantic fetch and publish endpoint counts must match: "
                f"fetch={self.endpoints!r} publish={self.publish_endpoints!r}"
            )
        common = {"timeout_ms": timeout_ms, "api_token": api_token}
        self._fetch_client_kwargs = [
            {"host": endpoint_host, "port": endpoint_port, **common}
            for endpoint_host, endpoint_port in self.endpoints
        ]
        self._publisher_client_kwargs = [
            {"host": endpoint_host, "port": endpoint_port, **common}
            for endpoint_host, endpoint_port in self.publish_endpoints
        ]
        self._fetch_clients: list[_ZmqRpcClient | None] = [None] * len(self.endpoints)
        self._fetch_executors = [
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"gr00t-semantic-fetch-{shard}"
            )
            for shard in range(len(self.endpoints))
        ]
        self._publisher_clients: list[_ZmqRpcClient] | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gr00t-semantic-publish"
        )
        self._publish_future: concurrent.futures.Future | None = None
        self._queued_publish: tuple[BatchFeature, dict[str, Any]] | None = None
        self._latest_fetch_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gr00t-semantic-latest"
        )
        self._latest_fetch_future: concurrent.futures.Future | None = None
        self._queued_latest_fetch: dict[str, Any] | None = None
        self._latest_fetch_replaced = 0
        self.last_metrics: dict[str, float] = {}
        self.last_response_metadata: dict[str, Any] = {}

    @staticmethod
    def _cpu_snapshot(backbone_inputs: BatchFeature) -> BatchFeature:
        return BatchFeature(
            data={
                key: value.detach().to(device="cpu").contiguous()
                for key, value in dict(backbone_inputs).items()
            }
        )

    def _shard_indices(self, env_ids: list[int]) -> list[list[int]]:
        shards = [[] for _ in self.endpoints]
        for row, env_id in enumerate(env_ids):
            shards[int(env_id) % len(shards)].append(row)
        return shards

    @staticmethod
    def _slice_batch(
        batch: BatchFeature, indices: list[int], batch_size: int
    ) -> BatchFeature:
        data = dict(batch)
        sliced = {
            key: value[indices]
            for key, value in data.items()
            if value.shape[0] == batch_size
        }
        handled = {key for key, value in data.items() if value.shape[0] == batch_size}

        # Qwen3-VL concatenates all image patches along dimension zero. Grid
        # metadata identifies the complete patch ranges belonging to each env.
        image_grid = data.get("image_grid_thw")
        pixel_values = data.get("pixel_values")
        if image_grid is not None and pixel_values is not None:
            if image_grid.shape[0] % batch_size != 0:
                raise ValueError(
                    "image_grid_thw rows are not divisible by env batch size"
                )
            images_per_env = image_grid.shape[0] // batch_size
            image_rows = [
                row
                for env_row in indices
                for row in range(
                    env_row * images_per_env, (env_row + 1) * images_per_env
                )
            ]
            patch_counts = image_grid.to(torch.int64).prod(dim=1).tolist()
            offsets = [0]
            for count in patch_counts:
                offsets.append(offsets[-1] + int(count))
            if offsets[-1] != pixel_values.shape[0]:
                raise ValueError(
                    "pixel_values rows do not match image_grid_thw patch count: "
                    f"pixels={pixel_values.shape[0]} grid_patches={offsets[-1]}"
                )
            patch_rows = [
                row
                for image_row in image_rows
                for row in range(offsets[image_row], offsets[image_row + 1])
            ]
            sliced["image_grid_thw"] = image_grid[image_rows]
            sliced["pixel_values"] = pixel_values[patch_rows]
            handled.update(("image_grid_thw", "pixel_values"))

        unknown = sorted(set(data) - handled)
        if unknown:
            raise ValueError(
                f"Cannot shard semantic backbone fields without batch layout: {unknown}"
            )
        return BatchFeature(data=sliced)

    @staticmethod
    def _slice_metadata(
        metadata: dict[str, Any], indices: list[int], batch_size: int
    ) -> dict[str, Any]:
        sliced = {}
        for key, value in metadata.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                sliced[key] = value[indices]
            elif (
                isinstance(value, np.ndarray)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                sliced[key] = value[indices]
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                sliced[key] = [value[index] for index in indices]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _merge_outputs(
        parts: list[tuple[list[int], BatchFeature]], batch_size: int
    ) -> BatchFeature:
        if not parts:
            raise RuntimeError("No semantic shard returned an output")
        keys = tuple(dict(parts[0][1]).keys())
        merged: dict[str, torch.Tensor] = {}
        for key in keys:
            tensors = [dict(outputs)[key] for _, outputs in parts]
            max_shape = tuple(
                max(tensor.shape[dim] for tensor in tensors)
                for dim in range(1, tensors[0].ndim)
            )
            target = torch.zeros(
                (batch_size, *max_shape),
                dtype=tensors[0].dtype,
                device=tensors[0].device,
            )
            for (indices, _), tensor in zip(parts, tensors, strict=True):
                slices = (slice(None),) + tuple(
                    slice(0, size) for size in tensor.shape[1:]
                )
                padded = torch.zeros(
                    (len(indices), *max_shape),
                    dtype=tensor.dtype,
                    device=tensor.device,
                )
                padded[slices] = tensor
                target[
                    torch.as_tensor(indices, device=target.device, dtype=torch.long)
                ] = padded
            merged[key] = target
        return BatchFeature(data=merged)

    @staticmethod
    def _merge_metadata(
        shard_results: list[tuple[list[int], dict[str, Any]]], batch_size: int
    ) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        keys = {key for _, metadata in shard_results for key in metadata}
        for key in keys:
            ordered: list[Any] = [None] * batch_size
            is_batched = True
            for indices, metadata in shard_results:
                value = metadata.get(key)
                if not isinstance(value, (list, tuple)) or len(value) != len(indices):
                    is_batched = False
                    break
                for row, item in zip(indices, value, strict=True):
                    ordered[row] = item
            if is_batched:
                merged[key] = ordered
        return merged

    def _publish_worker(
        self, backbone_inputs: BatchFeature, metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if self._publisher_clients is None:
            self._publisher_clients = [
                _ZmqRpcClient(**kwargs) for kwargs in self._publisher_client_kwargs
            ]
        env_ids = [int(value) for value in metadata["env_ids"]]
        batch_size = len(env_ids)
        responses = []
        for shard, indices in enumerate(self._shard_indices(env_ids)):
            if not indices:
                continue
            responses.append(
                self._publisher_clients[shard].call_endpoint(
                    "publish_observations",
                    {
                        "backbone_inputs": batch_feature_to_payload(
                            self._slice_batch(backbone_inputs, indices, batch_size)
                        ),
                        "metadata": self._slice_metadata(metadata, indices, batch_size),
                    },
                )
            )
        return responses

    def _launch_publish(
        self, backbone_inputs: BatchFeature, metadata: dict[str, Any]
    ) -> None:
        self._publish_future = self._executor.submit(
            self._publish_worker, backbone_inputs, metadata
        )

    def publish(self, backbone_inputs: BatchFeature, metadata: dict[str, Any]) -> None:
        snapshot = self._cpu_snapshot(backbone_inputs)
        if self._publish_future is not None and self._publish_future.done():
            self._publish_future.result()
            self._publish_future = None
        if self._publish_future is None:
            self._launch_publish(snapshot, dict(metadata))
        else:
            self._queued_publish = (snapshot, dict(metadata))

    def _poll_publisher(self) -> None:
        if self._publish_future is None or not self._publish_future.done():
            return
        self._publish_future.result()
        self._publish_future = None
        queued = self._queued_publish
        self._queued_publish = None
        if queued is not None:
            self._launch_publish(*queued)

    def _call_fetch_shard(self, shard: int, request: dict[str, Any]) -> Any:
        client = self._fetch_clients[shard]
        if client is None:
            client = _ZmqRpcClient(**self._fetch_client_kwargs[shard])
            self._fetch_clients[shard] = client
        return client.call_endpoint("fetch_latest", request)

    def _call_fetch_exact_shard(self, shard: int, request: dict[str, Any]) -> Any:
        client = self._fetch_clients[shard]
        if client is None:
            client = _ZmqRpcClient(**self._fetch_client_kwargs[shard])
            self._fetch_clients[shard] = client
        return client.call_endpoint("fetch_exact", request)

    def _close_fetch_shard(self, shard: int) -> None:
        client = self._fetch_clients[shard]
        if client is not None:
            client.close()
            self._fetch_clients[shard] = None

    def _fetch_latest_response(
        self,
        *,
        env_ids: list[int],
        episode_generations: list[int],
        current_frame_ids: list[int],
        wait_for_initial: bool,
        device: torch.device | str,
        floating_dtype: torch.dtype,
        request_delay_s: float = 0.0,
        max_wait_ms: float | None = None,
    ) -> tuple[BatchFeature, dict[str, Any], dict[str, float]] | None:
        started = time.perf_counter()
        if request_delay_s > 0:
            time.sleep(float(request_delay_s))
        batch_size = len(env_ids)
        shard_indices = self._shard_indices(env_ids)
        while True:
            pending_responses = []
            for shard, indices in enumerate(shard_indices):
                if not indices:
                    continue
                request = {
                    "env_ids": [env_ids[index] for index in indices],
                    "episode_generations": [
                        episode_generations[index] for index in indices
                    ],
                    "current_frame_ids": [
                        current_frame_ids[index] for index in indices
                    ],
                    "target_age_frames": self.fetch_target_age_frames,
                    "max_wait_ms": (
                        self.fetch_max_wait_ms
                        if max_wait_ms is None
                        else max(0.0, float(max_wait_ms))
                    ),
                }
                future = self._fetch_executors[shard].submit(
                    self._call_fetch_shard, shard, request
                )
                pending_responses.append((indices, future))
            responses = [
                (indices, future.result()) for indices, future in pending_responses
            ]
            missing: list[int] = []
            for _, response in responses:
                if not response.get("ready", False):
                    missing.extend(response.get("missing_env_ids", []))
            if not missing:
                break
            if not wait_for_initial:
                return None
            time.sleep(0.001)

        output_parts = [
            (
                indices,
                payload_to_batch_feature(
                    response["backbone_outputs"],
                    device=device,
                    floating_dtype=floating_dtype,
                ),
            )
            for indices, response in responses
        ]
        metadata_parts = [
            (indices, dict(response.get("metadata", {})))
            for indices, response in responses
        ]
        outputs = self._merge_outputs(output_parts, batch_size)
        outputs = dequantize_semantic_transport(outputs, floating_dtype)
        response_metadata = self._merge_metadata(metadata_parts, batch_size)
        metrics_parts = [
            (len(indices), dict(response.get("metrics", {})))
            for indices, response in responses
        ]
        total_rows = max(1, sum(size for size, _ in metrics_parts))
        metrics = {
            "semantic_server_cache_entries": float(
                sum(
                    metrics.get("semantic_server_cache_entries", 0.0)
                    for _, metrics in metrics_parts
                )
            ),
            "semantic_server_pending_batches": float(
                sum(
                    metrics.get("semantic_server_pending_batches", 0.0)
                    for _, metrics in metrics_parts
                )
            ),
            "semantic_server_batch_size": float(
                sum(
                    metrics.get("semantic_server_batch_size", 0.0)
                    for _, metrics in metrics_parts
                )
            ),
            "semantic_server_age_ms_mean": float(
                sum(
                    size * metrics.get("semantic_server_age_ms_mean", 0.0)
                    for size, metrics in metrics_parts
                )
                / total_rows
            ),
            "semantic_server_shards": float(len(responses)),
            "semantic_cache_fetch_ms": (time.perf_counter() - started) * 1000.0,
        }
        return outputs, response_metadata, metrics

    def fetch_latest(
        self,
        *,
        env_ids: list[int],
        episode_generations: list[int],
        current_frame_ids: list[int],
        wait_for_initial: bool,
        device: torch.device | str,
        floating_dtype: torch.dtype,
    ) -> tuple[BatchFeature, dict[str, Any]]:
        self._poll_publisher()
        response = self._fetch_latest_response(
            env_ids=env_ids,
            episode_generations=episode_generations,
            current_frame_ids=current_frame_ids,
            wait_for_initial=wait_for_initial,
            device=device,
            floating_dtype=floating_dtype,
        )
        if response is None:
            raise RuntimeError("Semantic cache is not ready")
        outputs, metadata, metrics = response
        self.last_response_metadata = metadata
        self.last_metrics = metrics
        return outputs, metadata

    def _launch_latest_fetch(self, request: dict[str, Any]) -> None:
        self._latest_fetch_future = self._latest_fetch_executor.submit(
            self._fetch_latest_response, **request
        )

    def submit_latest(
        self,
        *,
        env_ids: list[int],
        episode_generations: list[int],
        current_frame_ids: list[int],
        floating_dtype: torch.dtype,
        request_delay_s: float = 0.0,
        max_wait_ms: float | None = None,
    ) -> None:
        """Start a latest-only CPU fetch without blocking the control thread."""
        self._poll_publisher()
        request = {
            "env_ids": list(env_ids),
            "episode_generations": list(episode_generations),
            "current_frame_ids": list(current_frame_ids),
            "wait_for_initial": False,
            "device": "cpu",
            "floating_dtype": floating_dtype,
            "request_delay_s": max(0.0, float(request_delay_s)),
            "max_wait_ms": max_wait_ms,
        }
        if self._latest_fetch_future is None:
            self._launch_latest_fetch(request)
            return
        self._queued_latest_fetch = request
        self._latest_fetch_replaced += 1

    def _consume_latest_fetch(
        self,
        *,
        device: torch.device | str,
        floating_dtype: torch.dtype,
        wait: bool,
        timeout_ms: float | None = None,
    ) -> tuple[BatchFeature, dict[str, Any]] | None:
        future = self._latest_fetch_future
        if future is None or (not wait and not future.done()):
            return None
        wait_started = time.perf_counter()
        try:
            response = future.result(
                timeout=(
                    None
                    if timeout_ms is None
                    else max(0.0, float(timeout_ms)) / 1000.0
                )
            )
        except concurrent.futures.TimeoutError:
            foreground_wait_ms = (time.perf_counter() - wait_started) * 1000.0
            self.last_metrics = {
                "semantic_cache_foreground_wait_ms": foreground_wait_ms,
                "semantic_cache_fetch_replaced": float(self._latest_fetch_replaced),
                "semantic_cache_ready": 0.0,
                "semantic_cache_foreground_timeout": 1.0,
            }
            return None
        foreground_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        self._latest_fetch_future = None
        queued = self._queued_latest_fetch
        self._queued_latest_fetch = None
        if queued is not None:
            self._launch_latest_fetch(queued)
        if response is None:
            self.last_metrics = {
                "semantic_cache_foreground_wait_ms": foreground_wait_ms,
                "semantic_cache_fetch_replaced": float(self._latest_fetch_replaced),
                "semantic_cache_ready": 0.0,
            }
            return None
        outputs, metadata, metrics = response
        outputs = BatchFeature(
            data={
                key: value.to(
                    device=device,
                    dtype=(
                        floating_dtype
                        if torch.is_floating_point(value)
                        else value.dtype
                    ),
                )
                for key, value in dict(outputs).items()
            }
        )
        metrics["semantic_cache_foreground_wait_ms"] = foreground_wait_ms
        metrics["semantic_cache_fetch_replaced"] = float(self._latest_fetch_replaced)
        self.last_response_metadata = metadata
        self.last_metrics = metrics
        return outputs, metadata

    def poll_latest(
        self,
        *,
        device: torch.device | str,
        floating_dtype: torch.dtype,
    ) -> tuple[BatchFeature, dict[str, Any]] | None:
        return self._consume_latest_fetch(
            device=device, floating_dtype=floating_dtype, wait=False
        )

    def wait_latest(
        self,
        *,
        device: torch.device | str,
        floating_dtype: torch.dtype,
        timeout_ms: float | None = None,
    ) -> tuple[BatchFeature, dict[str, Any]] | None:
        return self._consume_latest_fetch(
            device=device,
            floating_dtype=floating_dtype,
            wait=True,
            timeout_ms=timeout_ms,
        )

    def fetch_exact(
        self,
        *,
        env_ids: list[int],
        episode_generations: list[int],
        source_frame_ids: list[int],
        max_wait_ms: float,
        device: torch.device | str,
        floating_dtype: torch.dtype,
    ) -> tuple[BatchFeature, dict[str, Any]] | None:
        """Late-join exact semantic packets with rollout samples by absolute frame."""
        self._poll_publisher()
        batch_size = len(env_ids)
        pending_responses = []
        for shard, indices in enumerate(self._shard_indices(env_ids)):
            if not indices:
                continue
            request = {
                "env_ids": [env_ids[index] for index in indices],
                "episode_generations": [
                    episode_generations[index] for index in indices
                ],
                "source_frame_ids": [source_frame_ids[index] for index in indices],
                "max_wait_ms": float(max_wait_ms),
            }
            future = self._fetch_executors[shard].submit(
                self._call_fetch_exact_shard, shard, request
            )
            pending_responses.append((indices, future))
        responses = [
            (indices, future.result()) for indices, future in pending_responses
        ]
        if any(not response.get("ready", False) for _, response in responses):
            return None
        output_parts = [
            (
                indices,
                payload_to_batch_feature(
                    response["backbone_outputs"],
                    device=device,
                    floating_dtype=floating_dtype,
                ),
            )
            for indices, response in responses
        ]
        metadata_parts = [
            (indices, dict(response.get("metadata", {})))
            for indices, response in responses
        ]
        outputs = dequantize_semantic_transport(
            self._merge_outputs(output_parts, batch_size), floating_dtype
        )
        return outputs, self._merge_metadata(metadata_parts, batch_size)

    def bootstrap_publish(
        self, backbone_inputs: BatchFeature, metadata: dict[str, Any]
    ) -> None:
        self._executor.submit(
            self._publish_worker,
            self._cpu_snapshot(backbone_inputs),
            dict(metadata),
        ).result()

    def close(self) -> None:
        if self._publish_future is not None:
            self._publish_future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self._latest_fetch_future is not None:
            self._latest_fetch_future.cancel()
        self._latest_fetch_executor.shutdown(wait=True, cancel_futures=True)
        for shard, executor in enumerate(self._fetch_executors):
            executor.submit(self._close_fetch_shard, shard).result()
            executor.shutdown(wait=True, cancel_futures=True)
        if self._publisher_clients is not None:
            for client in self._publisher_clients:
                client.close()


class Gr00tN1d7RawObservationPublisher:
    """Latest-only raw publisher sharded by environment."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | str = 6666,
        timeout_ms: int = 120000,
        api_token: str | None = None,
    ):
        self.endpoints = _parse_semantic_endpoints(host, port)
        self._client_kwargs = [
            {
                "host": endpoint_host,
                "port": endpoint_port,
                "timeout_ms": timeout_ms,
                "api_token": api_token,
            }
            for endpoint_host, endpoint_port in self.endpoints
        ]
        self._clients: list[_ZmqRpcClient | None] = [None] * len(self.endpoints)
        self._shard_executors = [
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"gr00t-midchunk-semantic-shard-{shard}",
            )
            for shard in range(len(self.endpoints))
        ]
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gr00t-midchunk-semantic"
        )
        self._lock = threading.RLock()
        self._future: concurrent.futures.Future | None = None
        self._queued: tuple[dict[str, Any], dict[str, Any]] | None = None
        self._error: BaseException | None = None
        self._closed = False
        self.replaced_count = 0

    @staticmethod
    def _slice_nested(value: Any, indices: list[int], batch_size: int) -> Any:
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value[indices]
            return value
        if isinstance(value, np.ndarray):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value[indices]
            return value
        if isinstance(value, dict):
            return {
                key: Gr00tN1d7RawObservationPublisher._slice_nested(
                    item, indices, batch_size
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            if len(value) == batch_size:
                return [value[index] for index in indices]
            return [
                Gr00tN1d7RawObservationPublisher._slice_nested(
                    item, indices, batch_size
                )
                for item in value
            ]
        if isinstance(value, tuple):
            if len(value) == batch_size:
                return tuple(value[index] for index in indices)
            return tuple(
                Gr00tN1d7RawObservationPublisher._slice_nested(
                    item, indices, batch_size
                )
                for item in value
            )
        return value

    def _call_shard(self, shard: int, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._clients[shard]
        if client is None:
            client = _ZmqRpcClient(**self._client_kwargs[shard])
            self._clients[shard] = client
        return client.call_endpoint("publish_raw_observations", payload)

    def _close_shard(self, shard: int) -> None:
        client = self._clients[shard]
        if client is not None:
            client.close()
            self._clients[shard] = None

    def _worker(
        self, observation: dict[str, Any], metadata: dict[str, Any]
    ) -> list[dict[str, Any]]:
        env_ids = [int(value) for value in metadata["env_ids"]]
        batch_size = len(env_ids)
        shard_indices = [[] for _ in self.endpoints]
        for row, env_id in enumerate(env_ids):
            shard_indices[env_id % len(shard_indices)].append(row)
        futures = []
        for shard, indices in enumerate(shard_indices):
            if not indices:
                continue
            futures.append(
                self._shard_executors[shard].submit(
                    self._call_shard,
                    shard,
                    {
                        "observation": self._slice_nested(
                            observation, indices, batch_size
                        ),
                        "metadata": self._slice_nested(metadata, indices, batch_size),
                    },
                )
            )
        return [future.result() for future in futures]

    def _launch_locked(
        self, observation: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        future = self._executor.submit(self._worker, observation, metadata)
        self._future = future
        future.add_done_callback(self._publish_done)

    def _publish_done(self, future: concurrent.futures.Future) -> None:
        cancelled = future.cancelled()
        error = None if cancelled else future.exception()
        with self._lock:
            if self._future is not future:
                return
            self._future = None
            if self._closed or cancelled:
                return
            if error is not None:
                self._error = error
                self._queued = None
                return
            queued = self._queued
            self._queued = None
            if queued is not None and not self._closed:
                self._launch_locked(*queued)

    def publish(self, observation: dict[str, Any], metadata: dict[str, Any]) -> None:
        snapshot = _snapshot_nested_cpu(observation)
        with self._lock:
            if self._closed:
                raise RuntimeError("Raw semantic publisher is closed")
            if self._error is not None:
                raise RuntimeError("Raw semantic publish failed") from self._error
            if self._future is None:
                self._launch_locked(snapshot, dict(metadata))
                return
            self._queued = (snapshot, dict(metadata))
            self.replaced_count += 1

    def poll(self) -> None:
        with self._lock:
            if self._error is not None:
                raise RuntimeError("Raw semantic publish failed") from self._error

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._queued = None
            future = self._future
        if future is not None:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)
        for shard, executor in enumerate(self._shard_executors):
            executor.submit(self._close_shard, shard).result()
            executor.shutdown(wait=True, cancel_futures=True)


class Gr00tN1d7AsyncSemanticBackboneClient:
    """Single-flight background wrapper around the synchronous semantic RPC client.

    ZMQ sockets are created and used exclusively by the worker thread. The rollout
    thread only copies an observation snapshot to CPU, submits it, and polls for a
    completed semantic packet without waiting for the VLM forward.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int | str = 6666,
        timeout_ms: int = 60000,
        api_token: str | None = None,
    ):
        self._client_kwargs = {
            "host": host,
            "port": port,
            "timeout_ms": timeout_ms,
            "api_token": api_token,
        }
        self.endpoints = _parse_semantic_endpoints(host, port)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="gr00t-semantic-rpc"
        )
        self._worker_client: Gr00tN1d7SemanticBackboneClient | None = None
        self._future: concurrent.futures.Future | None = None
        self._metadata: dict[str, Any] | None = None
        self._queued: tuple[BatchFeature, dict[str, Any]] | None = None
        self._latest_replaced_count = 0
        self.last_metrics: dict[str, float] = {}

    @staticmethod
    def _cpu_snapshot(backbone_inputs: BatchFeature) -> BatchFeature:
        return BatchFeature(
            data={
                key: value.detach().to(device="cpu").contiguous()
                for key, value in dict(backbone_inputs).items()
            }
        )

    def _encode_worker(
        self, backbone_inputs: BatchFeature, metadata: dict[str, Any]
    ) -> tuple[BatchFeature, dict[str, float], dict[str, Any]]:
        if self._worker_client is None:
            self._worker_client = Gr00tN1d7SemanticBackboneClient(**self._client_kwargs)
        outputs = self._worker_client.encode_backbone(
            backbone_inputs, metadata=metadata
        )
        return (
            outputs,
            dict(self._worker_client.last_metrics),
            dict(self._worker_client.last_response_metadata),
        )

    def _launch(self, snapshot: BatchFeature, metadata: dict[str, Any]) -> None:
        self._metadata = metadata
        self._future = self._executor.submit(self._encode_worker, snapshot, metadata)

    @property
    def pending(self) -> bool:
        return self._future is not None

    def submit(
        self, backbone_inputs: BatchFeature, metadata: dict[str, Any] | None = None
    ) -> bool:
        snapshot = self._cpu_snapshot(backbone_inputs)
        request_metadata = dict(metadata or {})
        request_metadata["semantic_request_started_wallclock_s"] = time.time()
        request_metadata["semantic_request_started_perf_s"] = time.perf_counter()
        if self._future is not None:
            self._queued = (snapshot, request_metadata)
            self._latest_replaced_count += 1
            return True
        self._launch(snapshot, request_metadata)
        return True

    def poll(
        self,
        *,
        device: torch.device | str | None = None,
        floating_dtype: torch.dtype | None = None,
    ) -> tuple[BatchFeature, dict[str, Any], dict[str, float]] | None:
        future = self._future
        if future is None or not future.done():
            return None
        metadata = dict(self._metadata or {})
        self._future = None
        self._metadata = None
        outputs, metrics, response_metadata = future.result()
        metadata.update(response_metadata)
        queued = self._queued
        self._queued = None
        if queued is not None:
            self._launch(*queued)
        outputs = BatchFeature(
            data={
                key: value.to(
                    device=device if device is not None else value.device,
                    dtype=(
                        floating_dtype
                        if floating_dtype is not None and torch.is_floating_point(value)
                        else value.dtype
                    ),
                )
                for key, value in dict(outputs).items()
            }
        )
        started = float(
            metadata.pop("semantic_request_started_perf_s", time.perf_counter())
        )
        metrics["semantic_async_end_to_end_ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        metrics["semantic_latest_replaced"] = float(self._latest_replaced_count)
        metrics["dyn_replace"] = float(self._latest_replaced_count)
        self.last_metrics = metrics
        return outputs, metadata, metrics

    def encode_backbone_blocking(
        self,
        backbone_inputs: BatchFeature,
        *,
        metadata: dict[str, Any] | None = None,
        device: torch.device | str | None = None,
        floating_dtype: torch.dtype | None = None,
    ) -> BatchFeature:
        while self._future is not None:
            pending_result = self.poll(device=device, floating_dtype=floating_dtype)
            if pending_result is None:
                time.sleep(0.001)
        if not self.submit(backbone_inputs, metadata=metadata):
            raise RuntimeError("Failed to submit blocking semantic bootstrap")
        while True:
            result = self.poll(device=device, floating_dtype=floating_dtype)
            if result is not None:
                outputs, _, metrics = result
                self.last_metrics = metrics
                return outputs
            time.sleep(0.001)

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)


class Gr00tN1d7SemanticBackbonePolicy:
    """PolicyServer-compatible object exposing only backbone encoding."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        local_files_only: bool | None = None,
        model_revision: str | None = None,
        cache_dir: str | None = None,
        load_bf16: bool | None = None,
        backbone_model_path: str | None = None,
        raw_preprocess_workers: int = 4,
        enable_raw_preprocessing: bool = True,
        fetch_pause_ms: float = 0.0,
        transport_quantization: str = "none",
        text_padding_tokens: int = 570,
        cache_history_size: int = 32,
    ):
        AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
        AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)

        loading_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if local_files_only is not None:
            loading_kwargs["local_files_only"] = local_files_only
        if model_revision is not None:
            loading_kwargs["revision"] = model_revision
        if cache_dir is not None:
            loading_kwargs["cache_dir"] = cache_dir

        config = Gr00tN1d7Config.from_pretrained(model_path, **loading_kwargs)
        if backbone_model_path is not None:
            config.backbone_pretrained_model_name_or_path = backbone_model_path
        if load_bf16 is not None:
            config.load_bf16 = bool(load_bf16)
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        self.torch_dtype = torch_dtype
        self.model = Gr00tN1d7.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            transformers_loading_kwargs=loading_kwargs,
        )
        self.model.to(device=self.device, dtype=torch_dtype)
        self.model.eval()
        processor_dir = Path(model_path) / "processor"
        if not processor_dir.is_dir():
            processor_dir = Path(model_path)
        with open(processor_dir / "processor_config.json") as file:
            processor_kwargs = json.load(file)["processor_kwargs"]
        with open(processor_dir / "statistics.json") as file:
            processor_kwargs["statistics"] = json.load(file)
        with open(processor_dir / "embodiment_id.json") as file:
            processor_kwargs["embodiment_id_mapping"] = json.load(file)
        processor_kwargs.setdefault("transformers_loading_kwargs", {})
        if local_files_only:
            processor_kwargs["transformers_loading_kwargs"]["local_files_only"] = True
        self.processor = Gr00tN1d7Processor(**processor_kwargs)
        self.embodiment_tag = EmbodimentTag("libero_sim")
        self.padding_value = max(0, int(text_padding_tokens))
        self.cache_history_size = max(1, int(cache_history_size))
        self.raw_preprocess_workers = max(1, int(raw_preprocess_workers))
        self.fetch_pause_s = max(0.0, float(fetch_pause_ms) / 1000.0)
        self.transport_quantization = str(transport_quantization).lower()
        if self.transport_quantization not in ("none", "int8"):
            raise ValueError(
                f"Unsupported semantic transport quantization: {transport_quantization}"
            )
        self.scheduler_pause_until_perf = 0.0
        self._forward_lock = threading.Lock()
        self._fetch_priority = threading.Event()
        self._freshness_demand = threading.Event()
        self._freshness_requirements: dict[int, tuple[int, int, int]] = {}
        self.scheduler_wakeup_callback = None
        self._raw_preprocess_executor = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=self.raw_preprocess_workers,
                thread_name_prefix="gr00t-semantic-preprocess",
            )
            if enable_raw_preprocessing
            else None
        )
        del self.model.action_head
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        logging.info("Unloaded unused DiT/action head from semantic server")
        self.latest_by_env: dict[tuple[str, int], dict[str, Any]] = {}
        self.pending_by_env: dict[int, dict[str, Any]] = {}
        self.pending_batches: dict[tuple[int, ...], dict[str, Any]] = {}
        self.pending_raw_batches: dict[tuple[int, ...], dict[str, Any]] = {}
        self._cache_lock = threading.RLock()
        self.semantic_cache_by_env: dict[int, dict[str, Any]] = {}
        self.semantic_cache_history_by_env: dict[int, deque[dict[str, Any]]] = {}
        self.last_scheduler_error: str | None = None
        self._semantic_version = 0
        logging.info(
            "Loaded GR00T N1.7 semantic backbone server from %s on %s",
            model_path,
            self.device,
        )

    def _store_semantic_cache_entry(
        self, env_id: int, cache_entry: dict[str, Any]
    ) -> None:
        # PPO can replay samples from an episode that ended later in the same
        # rollout. Keep those generations addressable until the bounded history
        # naturally evicts them.
        history = self.semantic_cache_history_by_env.setdefault(
            env_id, deque(maxlen=self.cache_history_size)
        )
        history.append(cache_entry)
        self.semantic_cache_by_env[env_id] = cache_entry

    @staticmethod
    def _input_signature(inputs: BatchFeature) -> tuple:
        return tuple(
            (key, tuple(value.shape[1:]), value.dtype)
            for key, value in sorted(dict(inputs).items())
        )

    @staticmethod
    def _canonicalize_text_inputs(
        inputs: dict[str, Any], padding_value: int
    ) -> dict[str, Any]:
        canonicalized = dict(inputs)
        for key in ("input_ids", "attention_mask"):
            tensor = canonicalized.get(key)
            if tensor is None or padding_value <= 0:
                continue
            if tensor.shape[-1] > padding_value:
                raise ValueError(
                    f"GR00T text field {key} length {tensor.shape[-1]} exceeds "
                    f"{padding_value}"
                )
            if tensor.shape[-1] < padding_value:
                tensor = torch.nn.functional.pad(
                    tensor, (0, padding_value - tensor.shape[-1]), value=0
                )
            canonicalized[key] = tensor
        return canonicalized

    def _prepare_backbone_input(self, inputs: dict[str, Any]) -> BatchFeature:
        inputs = dict(inputs)
        if "vlm_content" in inputs:
            vlm_content_list = inputs.pop("vlm_content")
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]
            prepared = self.model.collator(
                [{"vlm_content": content} for content in vlm_content_list]
            )["inputs"]
            inputs.update(prepared)
        backbone_inputs = self.model.backbone.prepare_input(inputs)
        return BatchFeature(
            data={
                key: value.to(
                    device=self.device,
                    dtype=(
                        self.torch_dtype
                        if torch.is_floating_point(value)
                        else value.dtype
                    ),
                )
                for key, value in dict(backbone_inputs).items()
            }
        )

    @torch.no_grad()
    def _prepare_raw_observation(self, observation: dict[str, Any]) -> BatchFeature:
        env_obs = dict(observation)
        env_obs["states"] = (
            torch.as_tensor(env_obs["states"]).to(torch.bfloat16).cpu().float()
        )
        env_obs["main_images"] = torch.as_tensor(env_obs["main_images"]).cpu()
        env_obs["wrist_images"] = torch.as_tensor(env_obs["wrist_images"]).cpu()
        converted = OBS_CONVERSION["libero"](env_obs)
        normalized = self.processor.process_observation(converted, self.embodiment_tag)
        normalized = {
            key: value.to(self.torch_dtype)
            if torch.is_tensor(value) and value.dtype == torch.float32
            else value
            for key, value in dict(normalized).items()
        }
        normalized = self._canonicalize_text_inputs(normalized, self.padding_value)
        return self._prepare_backbone_input(normalized)

    def _queue_observations(
        self, inputs: BatchFeature, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        with self._cache_lock:
            return self._queue_observations_locked(inputs, metadata)

    def _queue_observations_locked(
        self, inputs: BatchFeature, metadata: dict[str, Any]
    ) -> dict[str, Any]:
        batch_size = int(next(iter(dict(inputs).values())).shape[0])
        env_ids = torch.as_tensor(metadata["env_ids"]).reshape(-1).tolist()
        frame_ids = torch.as_tensor(metadata["frame_ids"]).reshape(-1).tolist()
        generations = (
            torch.as_tensor(metadata["episode_generations"]).reshape(-1).tolist()
        )
        wallclocks = (
            torch.as_tensor(
                metadata.get("observation_wallclock_s", [time.time()] * batch_size),
                dtype=torch.float64,
            )
            .reshape(-1)
            .tolist()
        )
        if not all(
            len(values) == batch_size
            for values in (env_ids, frame_ids, generations, wallclocks)
        ):
            raise ValueError("Semantic publish metadata does not match batch size")

        accepted = 0
        for row, env_id in enumerate(env_ids):
            env_id = int(env_id)
            generation = int(generations[row])
            frame_id = int(frame_ids[row])
            cached = self.semantic_cache_by_env.get(env_id)
            if cached is not None and (
                generation < cached["episode_generation"]
                or (
                    generation == cached["episode_generation"]
                    and frame_id < cached["source_frame_id"]
                )
            ):
                continue
            accepted += 1
        if accepted:
            batch_key = tuple(int(env_id) for env_id in env_ids)
            self.pending_batches[batch_key] = {
                "inputs": inputs,
                "signature": self._input_signature(inputs),
                "priority": int(metadata.get("semantic_priority", 1)),
                "env_ids": [int(value) for value in env_ids],
                "episode_generations": [int(value) for value in generations],
                "source_frame_ids": [int(value) for value in frame_ids],
                "source_wallclock_s": [float(value) for value in wallclocks],
                "published_wallclock_s": min(
                    (float(value) for value in wallclocks), default=time.time()
                ),
            }
            self._prune_superseded_pending_locked()
        return {
            "status": "queued",
            "accepted": accepted,
            "pending_batches": len(self.pending_batches),
        }

    def _prune_superseded_pending_locked(self) -> None:
        latest_by_env: dict[
            int, tuple[tuple[int, int, float], tuple[int, ...], int]
        ] = {}
        for batch_key, packet in self.pending_batches.items():
            for row, (env_id, generation, frame_id, wallclock_s) in enumerate(
                zip(
                    packet["env_ids"],
                    packet["episode_generations"],
                    packet["source_frame_ids"],
                    packet["source_wallclock_s"],
                    strict=True,
                )
            ):
                version = (int(generation), int(frame_id), float(wallclock_s))
                current = latest_by_env.get(int(env_id))
                if current is None or version >= current[0]:
                    latest_by_env[int(env_id)] = (version, batch_key, row)

        for batch_key, packet in list(self.pending_batches.items()):
            keep_rows = [
                row
                for row, env_id in enumerate(packet["env_ids"])
                if latest_by_env[int(env_id)][1:] == (batch_key, row)
            ]
            if not keep_rows:
                self.pending_batches.pop(batch_key, None)
                continue
            if len(keep_rows) == len(packet["env_ids"]):
                continue
            batch_size = len(packet["env_ids"])
            packet["inputs"] = Gr00tN1d7SemanticCacheClient._slice_batch(
                packet["inputs"], keep_rows, batch_size
            )
            for field in (
                "env_ids",
                "episode_generations",
                "source_frame_ids",
                "source_wallclock_s",
            ):
                values = packet[field]
                packet[field] = [values[row] for row in keep_rows]
            packet["published_wallclock_s"] = min(packet["source_wallclock_s"])

    def publish_observations(self, request: dict[str, Any]) -> dict[str, Any]:
        inputs = payload_to_batch_feature(request["backbone_inputs"])
        metadata = dict(request.get("metadata") or {})
        return self._queue_observations(inputs, metadata)

    def publish_raw_observations(self, request: dict[str, Any]) -> dict[str, Any]:
        if self._raw_preprocess_executor is None:
            raise RuntimeError(
                "Raw semantic preprocessing is disabled; publish through the "
                "preprocessing proxy"
            )
        metadata = dict(request.get("metadata") or {})
        env_ids = tuple(int(value) for value in metadata["env_ids"])
        submitted_perf = time.perf_counter()
        preprocess_future = self._raw_preprocess_executor.submit(
            self._prepare_raw_observation, request["observation"]
        )
        if self.scheduler_wakeup_callback is not None:
            preprocess_future.add_done_callback(
                lambda _: self.scheduler_wakeup_callback()
            )
        with self._cache_lock:
            previous = self.pending_raw_batches.get(env_ids)
            if previous is not None:
                previous["preprocess_future"].cancel()
            self.pending_raw_batches[env_ids] = {
                "observation": request["observation"],
                "metadata": metadata,
                "published_wallclock_s": time.time(),
                "preprocess_submitted_perf": submitted_perf,
                "preprocess_future": preprocess_future,
            }
            pending_count = len(self.pending_raw_batches)
        return {
            "status": "queued",
            "accepted": len(env_ids),
            "pending_raw_batches": pending_count,
        }

    def pending_request_counts(self) -> tuple[int, int]:
        with self._cache_lock:
            prepared = len(self.pending_batches)
            raw_total = len(self.pending_raw_batches)
            raw_ready = sum(
                packet["preprocess_future"].done()
                and not packet["preprocess_future"].cancelled()
                for packet in self.pending_raw_batches.values()
            )
        return prepared + raw_ready, prepared + raw_total

    def freshness_demand_active(self) -> bool:
        with self._cache_lock:
            requirements = getattr(self, "_freshness_requirements", None)
            if requirements is None:
                return self._freshness_demand.is_set()
            satisfied = []
            for env_id, requirement in requirements.items():
                generation, current_frame_id, target_age_frames = requirement
                cached = self.semantic_cache_by_env.get(env_id)
                if cached is None or cached["episode_generation"] != generation:
                    continue
                age_frames = current_frame_id - cached["source_frame_id"]
                if cached["source_frame_id"] <= current_frame_id and (
                    target_age_frames <= 0 or age_frames <= target_age_frames
                ):
                    satisfied.append(env_id)
            for env_id in satisfied:
                requirements.pop(env_id, None)
            active = bool(requirements)
            if not active:
                self._freshness_demand.clear()
            return active or self._freshness_demand.is_set()

    def _pending_packet_priority(
        self, packet: dict[str, Any]
    ) -> tuple[float, float, float, float]:
        requirements = getattr(self, "_freshness_requirements", {})
        urgent = any(int(env_id) in requirements for env_id in packet["env_ids"])
        explicit_priority = int(
            packet.get(
                "priority",
                packet.get("metadata", {}).get("semantic_priority", 1),
            )
        )
        oldest_completion = min(
            (
                self.semantic_cache_by_env.get(int(env_id), {}).get(
                    "completed_wallclock_s", 0.0
                )
                for env_id in packet["env_ids"]
            ),
            default=0.0,
        )
        return (
            float(explicit_priority),
            0.0 if urgent else 1.0,
            oldest_completion,
            -float(packet["published_wallclock_s"]),
        )

    def pending_env_counts(self) -> tuple[int, int]:
        with self._cache_lock:
            prepared_envs = {
                int(env_id)
                for packet in self.pending_batches.values()
                for env_id in packet["env_ids"]
            }
            raw_ready_envs = {
                int(env_id)
                for packet in self.pending_raw_batches.values()
                if packet["preprocess_future"].done()
                and not packet["preprocess_future"].cancelled()
                for env_id in packet["metadata"]["env_ids"]
            }
            raw_total_envs = {
                int(env_id)
                for packet in self.pending_raw_batches.values()
                for env_id in packet["metadata"]["env_ids"]
            }
        return (
            len(prepared_envs | raw_ready_envs),
            len(prepared_envs | raw_total_envs),
        )

    @staticmethod
    def _select_pending_packets(
        ordered: list[tuple[tuple[int, ...], dict[str, Any]]],
        *,
        max_requests: int,
        max_envs: int,
    ) -> list[tuple[tuple[int, ...], dict[str, Any]]]:
        selected = []
        selected_envs = 0
        for item in ordered:
            packet_envs = len(item[1]["env_ids"])
            if selected and max_envs > 0 and selected_envs + packet_envs > max_envs:
                break
            selected.append(item)
            selected_envs += packet_envs
            if len(selected) >= max_requests or (
                max_envs > 0 and selected_envs >= max_envs
            ):
                break
        return selected

    @torch.no_grad()
    def process_pending(self, max_batch_size: int, *, max_batch_envs: int = 0) -> int:
        if (
            self._fetch_priority.is_set()
            or time.perf_counter() < self.scheduler_pause_until_perf
        ):
            return 0
        with self._cache_lock:
            raw_selected = sorted(
                (
                    item
                    for item in self.pending_raw_batches.items()
                    if item[1]["preprocess_future"].done()
                    and not item[1]["preprocess_future"].cancelled()
                ),
                key=lambda item: self._pending_packet_priority(
                    {
                        **item[1],
                        "env_ids": item[1]["metadata"]["env_ids"],
                    }
                ),
            )[: max(1, int(max_batch_size))]
            for batch_key, _ in raw_selected:
                self.pending_raw_batches.pop(batch_key, None)
        raw_prep_started = time.perf_counter()
        raw_packets = [packet for _, packet in raw_selected]
        prepared_inputs = (
            packet["preprocess_future"].result() for packet in raw_packets
        )
        for inputs, packet in zip(prepared_inputs, raw_packets, strict=True):
            self._queue_observations(inputs, packet["metadata"])
        raw_prep_wait_ms = (time.perf_counter() - raw_prep_started) * 1000.0
        raw_prep_ms = max(
            (
                (time.perf_counter() - packet["preprocess_submitted_perf"]) * 1000.0
                for packet in raw_packets
            ),
            default=0.0,
        )

        with self._cache_lock:
            if not self.pending_batches:
                return 0
            ordered = sorted(
                self.pending_batches.items(),
                key=lambda item: self._pending_packet_priority(item[1]),
            )
            signature = ordered[0][1]["signature"]
            compatible = [
                (batch_key, packet)
                for batch_key, packet in ordered
                if packet["signature"] == signature
            ]
            selected = self._select_pending_packets(
                compatible,
                max_requests=max(1, int(max_batch_size)),
                max_envs=max(0, int(max_batch_envs)),
            )
            for batch_key, _ in selected:
                self.pending_batches.pop(batch_key, None)
        keys = tuple(dict(selected[0][1]["inputs"]).keys())
        try:
            merge_started = time.perf_counter()
            merged = BatchFeature(
                data={
                    key: torch.cat(
                        [packet["inputs"][key] for _, packet in selected], dim=0
                    ).to(
                        device=self.device,
                        dtype=(
                            self.torch_dtype
                            if torch.is_floating_point(selected[0][1]["inputs"][key])
                            else selected[0][1]["inputs"][key].dtype
                        ),
                    )
                    for key in keys
                }
            )
            merge_h2d_ms = (time.perf_counter() - merge_started) * 1000.0
            started = time.perf_counter()
            with self._forward_lock:
                if (
                    self._fetch_priority.is_set()
                    or time.perf_counter() < self.scheduler_pause_until_perf
                ):
                    with self._cache_lock:
                        for batch_key, packet in selected:
                            self.pending_batches.setdefault(batch_key, packet)
                    return 0
                outputs = self.model.backbone(merged)
        except Exception:
            with self._cache_lock:
                for batch_key, packet in selected:
                    self.pending_batches.setdefault(batch_key, packet)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise
        forward_ms = (time.perf_counter() - started) * 1000.0
        outputs = quantize_semantic_transport(outputs, self.transport_quantization)
        completed = time.time()
        output_row = 0
        merged_env_count = sum(len(packet["env_ids"]) for _, packet in selected)
        queue_age_ms = (
            max(
                (completed - packet["published_wallclock_s"] for _, packet in selected),
                default=0.0,
            )
            * 1000.0
        )
        logging.info(
            "Semantic batch requests=%d envs=%d raw_prep_ms=%.2f "
            "raw_prep_wait_ms=%.2f merge_h2d_ms=%.2f "
            "forward_ms=%.2f queue_age_ms=%.2f",
            len(selected),
            merged_env_count,
            raw_prep_ms,
            raw_prep_wait_ms,
            merge_h2d_ms,
            forward_ms,
            queue_age_ms,
        )
        for _, packet in selected:
            for row, env_id in enumerate(packet["env_ids"]):
                generation = packet["episode_generations"][row]
                frame_id = packet["source_frame_ids"][row]
                cached = self.semantic_cache_by_env.get(env_id)
                if cached is not None and (
                    generation < cached["episode_generation"]
                    or (
                        generation == cached["episode_generation"]
                        and frame_id < cached["source_frame_id"]
                    )
                ):
                    output_row += 1
                    continue
                self._semantic_version += 1
                cache_entry = {
                    "backbone_output": {
                        key: value[output_row : output_row + 1].detach().cpu()
                        for key, value in dict(outputs).items()
                    },
                    "episode_generation": generation,
                    "source_frame_id": frame_id,
                    "source_wallclock_s": packet["source_wallclock_s"][row],
                    "completed_wallclock_s": completed,
                    "version": self._semantic_version,
                    "forward_ms": forward_ms,
                    "batch_size": merged_env_count,
                }
                self._store_semantic_cache_entry(env_id, cache_entry)
                output_row += 1
        return merged_env_count

    @staticmethod
    def _stack_cached_outputs(packets: list[dict[str, Any]]) -> BatchFeature:
        keys = tuple(packets[0]["backbone_output"].keys())
        stacked = {}
        for key in keys:
            tensors = [packet["backbone_output"][key] for packet in packets]
            max_shape = tuple(
                max(tensor.shape[d] for tensor in tensors)
                for d in range(1, tensors[0].ndim)
            )
            padded = []
            for tensor in tensors:
                if tuple(tensor.shape[1:]) == max_shape:
                    padded.append(tensor)
                    continue
                target = torch.zeros(
                    (1, *max_shape), dtype=tensor.dtype, device=tensor.device
                )
                slices = (slice(None),) + tuple(
                    slice(0, size) for size in tensor.shape[1:]
                )
                target[slices] = tensor
                padded.append(target)
            stacked[key] = torch.cat(padded, dim=0)
        return BatchFeature(data=stacked)

    def fetch_latest(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.fetch_pause_s > 0:
            self._fetch_priority.set()
            try:
                with self._forward_lock:
                    self.scheduler_pause_until_perf = max(
                        self.scheduler_pause_until_perf,
                        time.perf_counter() + self.fetch_pause_s,
                    )
            finally:
                self._fetch_priority.clear()
        if self.last_scheduler_error is not None:
            return {"error": self.last_scheduler_error}
        env_ids = [int(value) for value in request["env_ids"]]
        generations = [
            int(value)
            for value in request.get("episode_generations", [0] * len(env_ids))
        ]
        current_frame_ids = [
            int(value)
            for value in request.get("current_frame_ids", [2**63 - 1] * len(env_ids))
        ]
        target_age_frames = max(0, int(request.get("target_age_frames", 0)))
        max_wait_s = max(0.0, float(request.get("max_wait_ms", 0.0)) / 1000.0)
        wait_started = time.perf_counter()
        wait_deadline = wait_started + max_wait_s
        while True:
            with self._cache_lock:
                packets = []
                for env_id, generation, current_frame_id in zip(
                    env_ids, generations, current_frame_ids, strict=True
                ):
                    candidates = [
                        packet
                        for packet in self.semantic_cache_history_by_env.get(env_id, ())
                        if packet["episode_generation"] == generation
                        and packet["source_frame_id"] <= current_frame_id
                    ]
                    packets.append(
                        max(candidates, key=lambda packet: packet["source_frame_id"])
                        if candidates
                        else None
                    )
            missing = [
                env_id
                for env_id, generation, packet in zip(
                    env_ids, generations, packets, strict=True
                )
                if packet is None or packet["episode_generation"] != generation
            ]
            stale = (
                [
                    env_id
                    for env_id, current_frame_id, packet in zip(
                        env_ids, current_frame_ids, packets, strict=True
                    )
                    if packet is not None
                    and current_frame_id - packet["source_frame_id"] > target_age_frames
                ]
                if target_age_frames > 0
                else []
            )
            if not missing and not stale:
                break
            with self._cache_lock:
                requirements = getattr(self, "_freshness_requirements", None)
                if requirements is not None:
                    urgent_env_ids = set(missing) | set(stale)
                    for env_id, generation, current_frame_id in zip(
                        env_ids, generations, current_frame_ids, strict=True
                    ):
                        if env_id in urgent_env_ids:
                            requirements[env_id] = (
                                generation,
                                current_frame_id,
                                target_age_frames,
                            )
                        else:
                            requirements.pop(env_id, None)
            if missing or time.perf_counter() < wait_deadline:
                self._freshness_demand.set()
                if self.scheduler_wakeup_callback is not None:
                    self.scheduler_wakeup_callback()
            if time.perf_counter() >= wait_deadline:
                break
            time.sleep(0.001)
        if not missing and not stale:
            with self._cache_lock:
                requirements = getattr(self, "_freshness_requirements", None)
                if requirements is not None:
                    for env_id in env_ids:
                        requirements.pop(env_id, None)
        freshness_wait_ms = (time.perf_counter() - wait_started) * 1000.0
        freshness_target_met = not missing and not stale
        if missing:
            return {"ready": False, "missing_env_ids": missing}
        packets = [packet for packet in packets if packet is not None]
        outputs = self._stack_cached_outputs(packets)
        now = time.time()
        return {
            "ready": True,
            "backbone_outputs": batch_feature_to_payload(outputs),
            "metadata": {
                "env_ids": env_ids,
                "source_frame_ids": [packet["source_frame_id"] for packet in packets],
                "episode_generations": [
                    packet["episode_generation"] for packet in packets
                ],
                "source_wallclock_s": [
                    packet["source_wallclock_s"] for packet in packets
                ],
                "completed_wallclock_s": [
                    packet["completed_wallclock_s"] for packet in packets
                ],
                "semantic_versions": [packet["version"] for packet in packets],
            },
            "metrics": {
                "semantic_server_cache_entries": float(len(self.semantic_cache_by_env)),
                "semantic_server_freshness_wait_ms": freshness_wait_ms,
                "semantic_server_freshness_target_met": float(freshness_target_met),
                "semantic_server_pending_batches": float(len(self.pending_batches)),
                "semantic_server_batch_size": float(
                    packets[-1]["batch_size"] if packets else 0
                ),
                "semantic_server_age_ms_mean": float(
                    np.mean(
                        [
                            max(0.0, now - packet["source_wallclock_s"]) * 1000.0
                            for packet in packets
                        ]
                    )
                    if packets
                    else 0.0
                ),
            },
        }

    def fetch_exact(self, request: dict[str, Any]) -> dict[str, Any]:
        """Return only packets whose generation and source frame exactly match."""
        if self.last_scheduler_error is not None:
            return {"error": self.last_scheduler_error}
        env_ids = [int(value) for value in request["env_ids"]]
        generations = [int(value) for value in request["episode_generations"]]
        source_frame_ids = [int(value) for value in request["source_frame_ids"]]
        max_wait_s = max(0.0, float(request.get("max_wait_ms", 0.0)) / 1000.0)
        deadline = time.perf_counter() + max_wait_s
        while True:
            with self._cache_lock:
                packets = []
                for env_id, generation, source_frame_id in zip(
                    env_ids, generations, source_frame_ids, strict=True
                ):
                    packets.append(
                        next(
                            (
                                packet
                                for packet in reversed(
                                    self.semantic_cache_history_by_env.get(env_id, ())
                                )
                                if packet["episode_generation"] == generation
                                and packet["source_frame_id"] == source_frame_id
                            ),
                            None,
                        )
                    )
            missing = [
                env_id
                for env_id, packet in zip(env_ids, packets, strict=True)
                if packet is None
            ]
            if not missing or time.perf_counter() >= deadline:
                break
            with self._cache_lock:
                for env_id, generation, source_frame_id in zip(
                    env_ids, generations, source_frame_ids, strict=True
                ):
                    if env_id in missing:
                        self._freshness_requirements[env_id] = (
                            generation,
                            source_frame_id,
                            1,
                        )
            self._freshness_demand.set()
            if self.scheduler_wakeup_callback is not None:
                self.scheduler_wakeup_callback()
            time.sleep(0.001)
        if missing:
            return {"ready": False, "missing_env_ids": missing}
        packets = [packet for packet in packets if packet is not None]
        outputs = self._stack_cached_outputs(packets)
        return {
            "ready": True,
            "backbone_outputs": batch_feature_to_payload(outputs),
            "metadata": {
                "source_frame_ids": [packet["source_frame_id"] for packet in packets],
                "episode_generations": [
                    packet["episode_generation"] for packet in packets
                ],
                "source_wallclock_s": [
                    packet["source_wallclock_s"] for packet in packets
                ],
                "completed_wallclock_s": [
                    packet["completed_wallclock_s"] for packet in packets
                ],
                "semantic_versions": [packet["version"] for packet in packets],
            },
        }

    def _format_response(
        self,
        outputs: BatchFeature,
        metadata: dict[str, Any],
        *,
        forward_ms: float,
        merged_batch_size: int,
        request_count: int,
    ) -> dict[str, Any]:
        batch_size = int(outputs["backbone_features"].shape[0])
        client_id = str(metadata.get("client_id", "anonymous"))
        env_slots = (
            torch.as_tensor(metadata.get("env_slots", range(batch_size)))
            .reshape(-1)
            .tolist()
        )
        frame_ids = (
            torch.as_tensor(metadata.get("frame_ids", [0] * batch_size))
            .reshape(-1)
            .tolist()
        )
        generations = (
            torch.as_tensor(metadata.get("episode_generations", [0] * batch_size))
            .reshape(-1)
            .tolist()
        )
        source_wallclock_s = float(metadata.get("observation_wallclock_s", time.time()))
        versions = []
        rejected = 0
        for row in range(batch_size):
            key = (client_id, int(env_slots[row]))
            generation = int(generations[row])
            frame_id = int(frame_ids[row])
            previous = self.latest_by_env.get(key)
            is_newer = previous is None or (
                generation > previous["episode_generation"]
                or (
                    generation == previous["episode_generation"]
                    and frame_id >= previous["source_frame_id"]
                )
            )
            if is_newer:
                self._semantic_version += 1
                version = self._semantic_version
                self.latest_by_env[key] = {
                    "backbone_output": {
                        name: value[row : row + 1].detach().cpu()
                        for name, value in dict(outputs).items()
                    },
                    "episode_generation": generation,
                    "source_frame_id": frame_id,
                    "source_wallclock_s": source_wallclock_s,
                    "completed_wallclock_s": time.time(),
                    "version": version,
                }
            else:
                rejected += 1
                version = int(previous["version"])
            versions.append(version)
        metrics = {
            "semantic_server_forward_ms": forward_ms,
            "semantic_server_batch_size": float(batch_size),
            "semantic_server_merged_batch_size": float(merged_batch_size),
            "semantic_server_batched_requests": float(request_count),
            "semantic_server_token_length": float(
                outputs["backbone_features"].shape[1]
            ),
            "semantic_server_cache_entries": float(len(self.latest_by_env)),
            "semantic_server_stale_rows_rejected": float(rejected),
            "dyn_cache_n": float(len(self.latest_by_env)),
            "dyn_stale": float(rejected),
        }
        return {
            "backbone_outputs": batch_feature_to_payload(outputs),
            "metrics": metrics,
            "metadata": {
                "semantic_versions": versions,
                "source_frame_ids": frame_ids,
                "episode_generations": generations,
                "source_wallclock_s": source_wallclock_s,
            },
        }

    @torch.no_grad()
    def encode_backbone_batch(
        self, requests: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not requests:
            return []
        inputs = [
            payload_to_batch_feature(
                request["backbone_inputs"],
                device=self.device,
                floating_dtype=self.torch_dtype,
            )
            for request in requests
        ]
        sizes = [int(next(iter(dict(batch).values())).shape[0]) for batch in inputs]
        keys = tuple(dict(inputs[0]).keys())
        if any(tuple(dict(batch).keys()) != keys for batch in inputs):
            raise ValueError(
                "Cannot batch semantic requests with different tensor keys"
            )
        merged = BatchFeature(
            data={
                key: torch.cat([batch[key] for batch in inputs], dim=0) for key in keys
            }
        )
        started = time.perf_counter()
        merged_outputs = self.model.backbone(merged)
        forward_ms = (time.perf_counter() - started) * 1000.0
        merged_outputs = quantize_semantic_transport(
            merged_outputs, self.transport_quantization
        )
        split_outputs = [
            BatchFeature(data=dict(zip(dict(merged_outputs), values, strict=True)))
            for values in zip(
                *[
                    torch.split(value, sizes, dim=0)
                    for value in dict(merged_outputs).values()
                ],
                strict=True,
            )
        ]
        return [
            self._format_response(
                outputs,
                dict(request.get("metadata") or {}),
                forward_ms=forward_ms,
                merged_batch_size=sum(sizes),
                request_count=len(requests),
            )
            for outputs, request in zip(split_outputs, requests, strict=True)
        ]

    @torch.no_grad()
    def encode_backbone(
        self,
        backbone_inputs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.encode_backbone_batch(
            [{"backbone_inputs": backbone_inputs, "metadata": metadata}]
        )[0]

    def get_action(self, *args, **kwargs):
        raise RuntimeError("This server exposes only encode_backbone, not get_action")

    def reset(self, *args, **kwargs) -> dict[str, str]:
        return {"status": "ok"}

    def get_modality_config(self) -> dict[str, Any]:
        return {}


def _parse_dtype(name: str) -> torch.dtype:
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve GR00T N1.7 backbone features.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=6666)
    parser.add_argument("--publish-port", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--batch-max-requests", type=int, default=8)
    parser.add_argument("--batch-target-requests", type=int, default=0)
    parser.add_argument("--batch-target-envs", type=int, default=0)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--bootstrap-target-envs", type=int, default=0)
    parser.add_argument("--bootstrap-wait-ms", type=float, default=30000.0)
    parser.add_argument("--rpc-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--raw-preprocess-workers", type=int, default=4)
    parser.add_argument("--text-padding-tokens", type=int, default=570)
    parser.add_argument("--cache-history-size", type=int, default=32)
    parser.add_argument("--disable-raw-preprocessing", action="store_true")
    parser.add_argument("--fetch-pause-ms", type=float, default=0.0)
    parser.add_argument(
        "--transport-quantization", default="none", choices=["none", "int8"]
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--backbone-model-path", default=None)
    parser.add_argument("--load-bf16", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    policy = Gr00tN1d7SemanticBackbonePolicy(
        args.model_path,
        device=args.device,
        torch_dtype=_parse_dtype(args.dtype),
        local_files_only=args.local_files_only or None,
        cache_dir=args.cache_dir,
        load_bf16=args.load_bf16 or None,
        backbone_model_path=args.backbone_model_path,
        raw_preprocess_workers=args.raw_preprocess_workers,
        enable_raw_preprocessing=not args.disable_raw_preprocessing,
        fetch_pause_ms=args.fetch_pause_ms,
        transport_quantization=args.transport_quantization,
        text_padding_tokens=args.text_padding_tokens,
        cache_history_size=args.cache_history_size,
    )
    with _ZmqRpcServer(
        policy,
        host=args.host,
        port=args.port,
        publish_port=args.publish_port,
        api_token=args.api_token,
        batch_max_requests=args.batch_max_requests,
        batch_target_requests=args.batch_target_requests,
        batch_target_envs=args.batch_target_envs,
        batch_wait_ms=args.batch_wait_ms,
        bootstrap_target_envs=args.bootstrap_target_envs,
        bootstrap_wait_ms=args.bootstrap_wait_ms,
        rpc_batch_wait_ms=args.rpc_batch_wait_ms,
    ) as server:
        server.run()


if __name__ == "__main__":
    main()
