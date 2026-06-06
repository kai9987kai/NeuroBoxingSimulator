import math

import torch

from neuroboxing import (
    ACTION_COUNT,
    Action,
    ActorCritic,
    BoxingEnv,
    FighterProfile,
    HeuristicAgent,
    TrainingConfig,
    load_policy,
    save_policy,
    simulate_match,
    train_self_play,
)


def test_observation_and_policy_shapes() -> None:
    environment = BoxingEnv(seed=1)
    observation = environment.observation(0)
    mask = environment.valid_action_mask(0)
    policy = ActorCritic()

    logits, value = policy(observation)

    assert observation.shape == (BoxingEnv.OBSERVATION_SIZE,)
    assert logits.shape == (ACTION_COUNT,)
    assert value.shape == ()
    assert mask.all()


def test_action_mask_blocks_expensive_actions_at_low_stamina() -> None:
    environment = BoxingEnv(seed=2)
    environment.fighters[0].stamina = 5.5

    mask = environment.valid_action_mask(0)

    assert bool(mask[Action.JAB])
    assert not bool(mask[Action.CROSS])
    assert not bool(mask[Action.HOOK])
    assert bool(mask[Action.GUARD])
    assert not bool(mask[Action.DODGE])
    assert bool(mask[Action.RECOVER])


def test_recover_restores_stamina_and_rewards_are_zero_sum() -> None:
    environment = BoxingEnv(seed=3)
    environment.fighters[0].stamina = 10.0

    first_reward, second_reward, done = environment.step(
        Action.RECOVER,
        Action.GUARD,
    )

    assert environment.fighters[0].stamina > 10.0
    assert math.isclose(first_reward + second_reward, 0.0, abs_tol=1e-9)
    assert not done


def test_seeded_matches_are_reproducible() -> None:
    profiles = (FighterProfile("A"), FighterProfile("B"))
    first = simulate_match(
        HeuristicAgent(seed=10),
        HeuristicAgent(seed=20),
        profiles=profiles,
        seed=30,
        max_exchanges=20,
    )
    second = simulate_match(
        HeuristicAgent(seed=10),
        HeuristicAgent(seed=20),
        profiles=profiles,
        seed=30,
        max_exchanges=20,
    )

    assert first.winner == second.winner
    assert first.method == second.method
    assert first.history == second.history


def test_policy_never_samples_a_masked_action() -> None:
    policy = ActorCritic()
    observation = torch.zeros(BoxingEnv.OBSERVATION_SIZE)
    mask = torch.tensor([False, False, False, False, False, True])

    sampled = {policy.act(observation, mask)[0] for _ in range(25)}

    assert sampled == {int(Action.RECOVER)}


def test_policy_round_trip(tmp_path) -> None:
    path = tmp_path / "policy.pt"
    policy = ActorCritic()
    save_policy(policy, path)

    loaded = load_policy(path)

    for expected, actual in zip(policy.parameters(), loaded.parameters()):
        assert torch.equal(expected, actual)


def test_short_training_run_produces_finite_metrics() -> None:
    policy = ActorCritic()
    config = TrainingConfig(
        episodes=4,
        batch_episodes=2,
        ppo_epochs=1,
        mini_batch_size=64,
        snapshot_interval=2,
        max_exchanges=8,
    )

    history = train_self_play(policy, config=config, seed=40)

    assert len(history) == 2
    assert all(
        math.isfinite(metrics[key])
        for metrics in history
        for key in ("mean_return", "policy_loss", "value_loss", "entropy")
    )
