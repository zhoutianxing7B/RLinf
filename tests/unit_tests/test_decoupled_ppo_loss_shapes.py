import torch

from rlinf.algorithms.losses import compute_ppo_actor_loss, compute_ppo_critic_loss


def test_actor_ratio_metrics_use_aligned_mask():
    logprobs = torch.zeros(2, 3, dtype=torch.float32, requires_grad=True)
    loss, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=torch.zeros(2, 3),
        advantages=torch.ones(2, 1),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    assert torch.isclose(metrics["actor/ratio"], torch.tensor(1.0))
    assert torch.isclose(
        metrics["actor/ratio_exp_logratio_gap"], torch.tensor(0.0)
    )
    loss.backward()


def test_critic_metrics_are_finite_for_constant_returns():
    values = torch.zeros(2, 3)
    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=torch.ones(2, 3),
        prev_values=values.clone(),
        value_clip=0.2,
        huber_delta=10.0,
        loss_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["critic/explained_variance"])


def test_actor_rejects_impossible_logprob_shape():
    try:
        compute_ppo_actor_loss(
            logprobs=torch.zeros(2, 3),
            old_logprobs=torch.zeros(2, 2),
            advantages=torch.ones(2),
            clip_ratio_low=0.2,
            clip_ratio_high=0.2,
        )
    except ValueError:
        return
    raise AssertionError("incompatible PPO logprob shapes were accepted")
