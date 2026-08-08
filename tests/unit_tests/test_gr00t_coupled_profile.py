from pathlib import Path

import yaml


CONFIG = (
    Path(__file__).parents[2]
    / "examples/embodiment/config/libero_spatial_ppo_gr00t_n1d7_coupled.yaml"
)


def test_coupled_profile_disables_semantic_transport():
    cfg = yaml.safe_load(CONFIG.read_text())
    rl_head = cfg["actor"]["model"]["rl_head_config"]
    assert rl_head["execution_mode"] == "coupled"
    assert rl_head["semantic_server_enabled"] is False
    assert rl_head["drop_local_backbone"] is False
    assert rl_head["dit_only_train"] is False
    assert cfg["rollout"]["model"]["rl_head_config"] == "${actor.model.rl_head_config}"
