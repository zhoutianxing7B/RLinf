# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import zmq
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.gr00t.gr00t_n1d7.semantic_server import (
    _ZmqRpcClient,
    batch_feature_to_payload,
)
from rlinf.models.embodiment.gr00t.simulation_io import OBS_CONVERSION


def _load_processor(
    model_path: str,
    *,
    local_files_only: bool,
) -> Gr00tN1d7Processor:
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
    processor = Gr00tN1d7Processor(**processor_kwargs)
    processor.eval()
    return processor


class Gr00tN1d7SemanticPreprocessProxy:
    """Move raw observation preprocessing out of the GPU backbone process."""

    def __init__(
        self,
        model_path: str,
        *,
        host: str,
        port: int,
        target_host: str,
        target_port: int,
        workers: int = 8,
        batch_max_requests: int = 12,
        batch_target_envs: int = 60,
        batch_wait_ms: float = 500.0,
        timeout_ms: int = 120000,
        api_token: str | None = None,
        local_files_only: bool = False,
        text_padding_tokens: int = 570,
    ):
        self.host = host
        self.port = int(port)
        self.target_host = target_host
        self.target_port = int(target_port)
        self.timeout_ms = int(timeout_ms)
        self.api_token = api_token
        self.batch_max_requests = max(1, int(batch_max_requests))
        self.batch_target_envs = max(0, int(batch_target_envs))
        self.batch_wait_ms = max(0.0, float(batch_wait_ms))
        self.processor = _load_processor(
            model_path,
            local_files_only=local_files_only,
        )
        self.embodiment_tag = EmbodimentTag("libero_sim")
        self.padding_value = max(0, int(text_padding_tokens))
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://{host}:{port}")
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(workers)),
            thread_name_prefix="gr00t-semantic-proxy",
        )
        self._thread_local = threading.local()
        self._pending_lock = threading.Lock()
        self._pending_condition = threading.Condition(self._pending_lock)
        self._pending_raw: dict[tuple[int, ...], dict[str, Any]] = {}
        self._pending_by_env: dict[tuple[int, ...], concurrent.futures.Future] = {}
        self.replaced_count = 0
        self._scheduler_thread: threading.Thread | None = None

    def _target_client(self) -> _ZmqRpcClient:
        client = getattr(self._thread_local, "target_client", None)
        if client is None:
            client = _ZmqRpcClient(
                host=self.target_host,
                port=self.target_port,
                timeout_ms=self.timeout_ms,
                api_token=self.api_token,
            )
            self._thread_local.target_client = client
        return client

    @staticmethod
    def _canonicalize_text_inputs(inputs: dict[str, Any], padding_value: int):
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
                    tensor,
                    (0, padding_value - tensor.shape[-1]),
                    value=0,
                )
            canonicalized[key] = tensor
        return canonicalized

    @torch.no_grad()
    def _prepare_raw_observation(self, observation: dict[str, Any]) -> BatchFeature:
        env_obs = dict(observation)
        env_obs["states"] = (
            torch.as_tensor(env_obs["states"]).to(torch.bfloat16).cpu().float()
        )
        env_obs["main_images"] = torch.as_tensor(env_obs["main_images"]).cpu()
        env_obs["wrist_images"] = torch.as_tensor(env_obs["wrist_images"]).cpu()
        converted = OBS_CONVERSION["libero"](env_obs)
        normalized = dict(
            self.processor.process_observation(converted, self.embodiment_tag)
        )
        vlm_content = normalized.pop("vlm_content", None)
        if vlm_content is not None:
            if not isinstance(vlm_content, list):
                vlm_content = [vlm_content]
            prepared = self.processor.collator(
                [{"vlm_content": content} for content in vlm_content]
            )["inputs"]
            normalized.update(prepared)
        normalized = {
            key: value.to(torch.bfloat16)
            if torch.is_tensor(value) and value.dtype == torch.float32
            else value
            for key, value in normalized.items()
        }
        normalized = self._canonicalize_text_inputs(
            normalized,
            self.padding_value,
        )
        return BatchFeature(data=normalized)

    def _forward_request(self, request: dict[str, Any]) -> dict[str, Any]:
        endpoint = request.get("endpoint")
        data = dict(request.get("data") or {})
        if endpoint == "publish_raw_observations":
            inputs = self._prepare_raw_observation(data["observation"])
            endpoint = "publish_observations"
            data = {
                "backbone_inputs": batch_feature_to_payload(inputs),
                "metadata": dict(data.get("metadata") or {}),
            }
        return self._target_client().call_endpoint(endpoint, data)

    @staticmethod
    def _concat_batch_values(values: list[Any]) -> Any:
        first = values[0]
        if torch.is_tensor(first):
            return torch.cat(values, dim=0)
        if isinstance(first, np.ndarray):
            return np.concatenate(values, axis=0)
        if isinstance(first, dict):
            return {
                key: Gr00tN1d7SemanticPreprocessProxy._concat_batch_values(
                    [value[key] for value in values]
                )
                for key in first
            }
        if isinstance(first, list):
            return [item for value in values for item in value]
        if isinstance(first, tuple):
            return tuple(item for value in values for item in value)
        if all(value == first for value in values[1:]):
            return first
        raise ValueError(f"Cannot merge semantic proxy values of type {type(first)!r}")

    @classmethod
    def _merge_raw_requests(
        cls, requests: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        data = [dict(request.get("data") or {}) for request in requests]
        observations = [dict(item["observation"]) for item in data]
        observation_keys = tuple(observations[0])
        if any(tuple(observation) != observation_keys for observation in observations):
            raise ValueError("Raw semantic observations have incompatible keys")
        merged_observation = {
            key: cls._concat_batch_values(
                [observation[key] for observation in observations]
            )
            for key in observation_keys
        }
        metadatas = [dict(item.get("metadata") or {}) for item in data]
        metadata_keys = set().union(*(metadata.keys() for metadata in metadatas))
        merged_metadata = {}
        for key in metadata_keys:
            values = [metadata[key] for metadata in metadatas if key in metadata]
            if key == "semantic_priority":
                merged_metadata[key] = min(int(value) for value in values)
            else:
                merged_metadata[key] = cls._concat_batch_values(values)
        return merged_observation, merged_metadata

    def _forward_raw_batch(self, packets: list[dict[str, Any]]) -> None:
        started = time.perf_counter()
        requests = [packet["request"] for packet in packets]
        observation, metadata = self._merge_raw_requests(requests)
        prep_started = time.perf_counter()
        inputs = self._prepare_raw_observation(observation)
        prep_ms = (time.perf_counter() - prep_started) * 1000.0
        forward_started = time.perf_counter()
        self._target_client().call_endpoint(
            "publish_observations",
            {
                "backbone_inputs": batch_feature_to_payload(inputs),
                "metadata": metadata,
            },
        )
        forward_ms = (time.perf_counter() - forward_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0
        queue_ms = (
            max((started - packet["submitted_perf"] for packet in packets), default=0.0)
            * 1000.0
        )
        logging.info(
            "Semantic proxy batch requests=%d envs=%d queue_ms=%.2f "
            "preprocess_ms=%.2f forward_ms=%.2f total_ms=%.2f replaced=%d",
            len(packets),
            len(metadata.get("env_ids", ())),
            queue_ms,
            prep_ms,
            forward_ms,
            total_ms,
            self.replaced_count,
        )

    def _take_raw_batch(self) -> list[dict[str, Any]]:
        with self._pending_condition:
            while self.running:
                packets = list(self._pending_raw.values())
                if packets:
                    env_count = sum(packet["env_count"] for packet in packets)
                    oldest_age_ms = (
                        time.perf_counter() - packets[0]["submitted_perf"]
                    ) * 1000.0
                    if (
                        self.batch_target_envs <= 0
                        or env_count >= self.batch_target_envs
                        or oldest_age_ms >= self.batch_wait_ms
                    ):
                        break
                    self._pending_condition.wait(
                        timeout=max(
                            0.001,
                            (self.batch_wait_ms - oldest_age_ms) / 1000.0,
                        )
                    )
                else:
                    self._pending_condition.wait(timeout=0.1)
            if not self.running:
                return []
            selected = []
            for env_ids in list(self._pending_raw):
                if len(selected) >= self.batch_max_requests:
                    break
                selected.append(self._pending_raw.pop(env_ids))
            return selected

    def _run_scheduler(self) -> None:
        while self.running:
            packets = self._take_raw_batch()
            if not packets:
                continue
            try:
                self._forward_raw_batch(packets)
            except Exception:
                logging.exception("Semantic proxy batched preprocessing failed")

    def _future_done(
        self,
        env_ids: tuple[int, ...],
        future: concurrent.futures.Future,
    ) -> None:
        with self._pending_lock:
            if self._pending_by_env.get(env_ids) is future:
                self._pending_by_env.pop(env_ids, None)
        try:
            future.result()
        except concurrent.futures.CancelledError:
            return
        except Exception:
            logging.exception("Semantic preprocess proxy forwarding failed")

    def _submit_forward(self, request: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(request.get("data", {}).get("metadata") or {})
        env_ids = tuple(int(value) for value in metadata.get("env_ids", ()))
        with self._pending_lock:
            previous = self._pending_by_env.get(env_ids)
            if previous is not None and previous.cancel():
                self.replaced_count += 1
            future = self._executor.submit(self._forward_request, request)
            self._pending_by_env[env_ids] = future
        future.add_done_callback(
            lambda completed, key=env_ids: self._future_done(key, completed)
        )
        return {
            "status": "queued",
            "accepted": len(env_ids),
            "proxy_pending": len(self._pending_by_env),
            "proxy_replaced": self.replaced_count,
        }

    def _submit_raw(self, request: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(request.get("data", {}).get("metadata") or {})
        env_ids = tuple(int(value) for value in metadata.get("env_ids", ()))
        with self._pending_condition:
            if env_ids in self._pending_raw:
                self._pending_raw.pop(env_ids)
                self.replaced_count += 1
            self._pending_raw[env_ids] = {
                "request": request,
                "env_count": len(env_ids),
                "submitted_perf": time.perf_counter(),
            }
            pending = len(self._pending_raw)
            pending_envs = sum(
                packet["env_count"] for packet in self._pending_raw.values()
            )
            self._pending_condition.notify()
        return {
            "status": "queued",
            "accepted": len(env_ids),
            "proxy_pending": pending,
            "proxy_pending_envs": pending_envs,
            "proxy_replaced": self.replaced_count,
        }

    def run(self) -> None:
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            name="gr00t-semantic-proxy-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()
        logging.info(
            "Semantic preprocess proxy listening on tcp://%s:%d -> tcp://%s:%d "
            "(batch_max_requests=%d, batch_target_envs=%d, batch_wait_ms=%.2f)",
            self.host,
            self.port,
            self.target_host,
            self.target_port,
            self.batch_max_requests,
            self.batch_target_envs,
            self.batch_wait_ms,
        )
        while self.running:
            frames = self.socket.recv_multipart()
            envelope, request = frames[:-1], pickle.loads(frames[-1])
            if (
                self.api_token is not None
                and request.get("api_token") != self.api_token
            ):
                result = {"error": "unauthorized"}
            elif request.get("endpoint") == "ping":
                result = {"status": "ok"}
            elif request.get("endpoint") == "kill":
                self.running = False
                result = {"status": "stopping"}
            elif request.get("endpoint") == "publish_raw_observations":
                result = self._submit_raw(request)
            elif request.get("endpoint") == "publish_observations":
                result = self._submit_forward(request)
            else:
                result = {"error": f"Unknown proxy endpoint: {request.get('endpoint')}"}
            self.socket.send_multipart(
                [
                    *envelope,
                    pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL),
                ]
            )

    def close(self) -> None:
        self.running = False
        with self._pending_condition:
            self._pending_condition.notify_all()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=5.0)
        self.socket.close(linger=0)
        self.context.term()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess GR00T observations before the semantic GPU server."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-max-requests", type=int, default=12)
    parser.add_argument("--batch-target-envs", type=int, default=60)
    parser.add_argument("--batch-wait-ms", type=float, default=500.0)
    parser.add_argument("--timeout-ms", type=int, default=120000)
    parser.add_argument("--api-token", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--text-padding-tokens", type=int, default=570)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    with Gr00tN1d7SemanticPreprocessProxy(
        args.model_path,
        host=args.host,
        port=args.port,
        target_host=args.target_host,
        target_port=args.target_port,
        workers=args.workers,
        batch_max_requests=args.batch_max_requests,
        batch_target_envs=args.batch_target_envs,
        batch_wait_ms=args.batch_wait_ms,
        timeout_ms=args.timeout_ms,
        api_token=args.api_token,
        local_files_only=args.local_files_only,
        text_padding_tokens=args.text_padding_tokens,
    ) as proxy:
        proxy.run()


if __name__ == "__main__":
    main()
