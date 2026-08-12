from omegaconf import OmegaConf

from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def _worker(dataset_type: str | None):
    worker = object.__new__(FSDPVlaSftWorker)
    data = {} if dataset_type is None else {"dataset_type": dataset_type}
    worker.cfg = OmegaConf.create(
        {
            "actor": {"model": {"model_type": "gr00t_n1d7"}},
            "data": data,
        }
    )
    return worker


def test_cached_semantic_action_sft_requires_explicit_dataset_type():
    assert not _worker(None)._uses_shared_semantic_action_dataset()
    assert not _worker("binary_image")._uses_shared_semantic_action_dataset()
    assert _worker(
        "shared_semantic_action"
    )._uses_shared_semantic_action_dataset()


def test_non_gr00t_model_never_uses_cached_semantic_action_dataset():
    worker = _worker("shared_semantic_action")
    worker.cfg.actor.model.model_type = "openpi"

    assert not worker._uses_shared_semantic_action_dataset()
