from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable, Protocol, Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Categorical


MAX_HEALTH = 100.0
MAX_STAMINA = 100.0


class Action(IntEnum):
    JAB = 0
    CROSS = 1
    HOOK = 2
    GUARD = 3
    DODGE = 4
    RECOVER = 5


@dataclass(frozen=True)
class ActionSpec:
    stamina_cost: float
    base_damage: float = 0.0
    accuracy: float = 0.0


ACTION_SPECS: tuple[ActionSpec, ...] = (
    ActionSpec(stamina_cost=5.0, base_damage=6.0, accuracy=0.78),
    ActionSpec(stamina_cost=9.0, base_damage=10.5, accuracy=0.64),
    ActionSpec(stamina_cost=12.0, base_damage=13.5, accuracy=0.55),
    ActionSpec(stamina_cost=4.0),
    ActionSpec(stamina_cost=6.0),
    ActionSpec(stamina_cost=0.0),
)
ATTACK_ACTIONS = frozenset((Action.JAB, Action.CROSS, Action.HOOK))
ACTION_COUNT = len(Action)


@dataclass(frozen=True)
class FighterProfile:
    name: str
    power: float = 1.0
    speed: float = 1.0
    defense: float = 1.0
    conditioning: float = 1.0


@dataclass
class FighterState:
    profile: FighterProfile
    health: float = MAX_HEALTH
    stamina: float = MAX_STAMINA
    score: float = 0.0
    punches_thrown: int = 0
    punches_landed: int = 0
    last_action: Action | None = None

    @property
    def accuracy(self) -> float:
        if self.punches_thrown == 0:
            return 0.0
        return self.punches_landed / self.punches_thrown


@dataclass(frozen=True)
class StepRecord:
    exchange: int
    health: tuple[float, float]
    stamina: tuple[float, float]
    score: tuple[float, float]
    actions: tuple[Action, Action]
    hits: tuple[bool, bool]


@dataclass
class MatchResult:
    fighters: tuple[FighterState, FighterState]
    winner: int | None
    method: str
    exchanges: int
    exchanges_per_round: int
    history: list[StepRecord] = field(default_factory=list)

    def summary(self) -> str:
        first, second = self.fighters
        if self.winner is None:
            outcome = f"Draw after {self.exchanges} exchanges"
        else:
            winner = self.fighters[self.winner]
            outcome = f"{winner.profile.name} wins by {self.method}"
        return (
            f"{outcome}. "
            f"Score {first.score:.1f}-{second.score:.1f}; "
            f"accuracy {first.accuracy:.1%}-{second.accuracy:.1%}."
        )


class Agent(Protocol):
    def choose_action(self, observation: Tensor, action_mask: Tensor) -> int:
        ...


class BoxingEnv:
    """A compact, simultaneous-action boxing game with normalized mechanics."""

    OBSERVATION_SIZE = 28

    def __init__(
        self,
        profiles: tuple[FighterProfile, FighterProfile] | None = None,
        *,
        seed: int = 0,
        max_exchanges: int = 60,
        exchanges_per_round: int = 20,
    ) -> None:
        if max_exchanges <= 0:
            raise ValueError("max_exchanges must be positive")
        self.rng = random.Random(seed)
        self.max_exchanges = max_exchanges
        self.exchanges_per_round = exchanges_per_round
        self.profiles = profiles or (
            FighterProfile("Boxer 1"),
            FighterProfile("Boxer 2"),
        )
        self.fighters = [FighterState(profile) for profile in self.profiles]
        self.exchange = 0
        self.done = False
        self.history: list[StepRecord] = []

    def observation(self, fighter_index: int) -> Tensor:
        own = self.fighters[fighter_index]
        opponent = self.fighters[1 - fighter_index]
        progress = self.exchange / self.max_exchanges
        score_difference = (own.score - opponent.score) / 50.0
        own_last = self._one_hot_action(own.last_action)
        opponent_last = self._one_hot_action(opponent.last_action)
        values = [
            own.health / MAX_HEALTH,
            own.stamina / MAX_STAMINA,
            own.score / 100.0,
            own.profile.power,
            own.profile.speed,
            own.profile.defense,
            own.profile.conditioning,
            opponent.health / MAX_HEALTH,
            opponent.stamina / MAX_STAMINA,
            opponent.score / 100.0,
            opponent.profile.power,
            opponent.profile.speed,
            opponent.profile.defense,
            opponent.profile.conditioning,
            progress,
            score_difference,
            *own_last,
            *opponent_last,
        ]
        return torch.tensor(values, dtype=torch.float32)

    @staticmethod
    def _one_hot_action(action: Action | None) -> list[float]:
        encoded = [0.0] * ACTION_COUNT
        if action is not None:
            encoded[int(action)] = 1.0
        return encoded

    def valid_action_mask(self, fighter_index: int) -> Tensor:
        stamina = self.fighters[fighter_index].stamina
        return torch.tensor(
            [stamina >= spec.stamina_cost for spec in ACTION_SPECS],
            dtype=torch.bool,
        )

    def step(self, first_action: int, second_action: int) -> tuple[float, float, bool]:
        if self.done:
            raise RuntimeError("The match is already complete")

        actions = (Action(first_action), Action(second_action))
        for index, action in enumerate(actions):
            if not self.valid_action_mask(index)[int(action)]:
                raise ValueError(
                    f"{action.name} is invalid with "
                    f"{self.fighters[index].stamina:.1f} stamina"
                )

        for fighter, action in zip(self.fighters, actions):
            cost = ACTION_SPECS[int(action)].stamina_cost / fighter.profile.conditioning
            fighter.stamina = max(0.0, fighter.stamina - cost)
            fighter.last_action = action
            if action in ATTACK_ACTIONS:
                fighter.punches_thrown += 1

        first_hit, damage_to_second = self._resolve_attack(0, 1, actions[0], actions[1])
        second_hit, damage_to_first = self._resolve_attack(1, 0, actions[1], actions[0])

        self.fighters[0].health = max(0.0, self.fighters[0].health - damage_to_first)
        self.fighters[1].health = max(0.0, self.fighters[1].health - damage_to_second)
        first_score = self._score_hit(0, first_hit, damage_to_second)
        second_score = self._score_hit(1, second_hit, damage_to_first)

        for fighter, action in zip(self.fighters, actions):
            recovery = 0.6 * fighter.profile.conditioning
            if action == Action.RECOVER:
                recovery += 12.0 * fighter.profile.conditioning
            elif action == Action.GUARD:
                recovery += 1.5 * fighter.profile.conditioning
            fighter.stamina = min(MAX_STAMINA, fighter.stamina + recovery)

        self.exchange += 1
        self.done = (
            any(fighter.health <= 0.0 for fighter in self.fighters)
            or self.exchange >= self.max_exchanges
        )

        reward = (damage_to_second - damage_to_first) / 20.0
        reward += (first_score - second_score) * 0.03
        if self.done:
            winner, _ = self._outcome()
            if winner == 0:
                reward += 1.0
            elif winner == 1:
                reward -= 1.0

        self.history.append(
            StepRecord(
                exchange=self.exchange,
                health=(self.fighters[0].health, self.fighters[1].health),
                stamina=(self.fighters[0].stamina, self.fighters[1].stamina),
                score=(self.fighters[0].score, self.fighters[1].score),
                actions=actions,
                hits=(first_hit, second_hit),
            )
        )
        return reward, -reward, self.done

    def _resolve_attack(
        self,
        attacker_index: int,
        defender_index: int,
        attack_action: Action,
        defense_action: Action,
    ) -> tuple[bool, float]:
        if attack_action not in ATTACK_ACTIONS:
            return False, 0.0

        attacker = self.fighters[attacker_index]
        defender = self.fighters[defender_index]
        spec = ACTION_SPECS[int(attack_action)]
        stamina_ratio = attacker.stamina / MAX_STAMINA
        skill_factor = _clamp(
            1.0 + 0.25 * (attacker.profile.speed - defender.profile.defense),
            0.78,
            1.22,
        )
        hit_probability = spec.accuracy * skill_factor * (0.82 + 0.18 * stamina_ratio)
        damage_multiplier = 1.0

        if defense_action == Action.GUARD:
            hit_probability *= 0.72
            damage_multiplier *= 0.48
        elif defense_action == Action.DODGE:
            dodge_factor = _clamp(
                0.58 - 0.12 * (defender.profile.speed - attacker.profile.speed),
                0.38,
                0.72,
            )
            hit_probability *= dodge_factor
            damage_multiplier *= 0.80
        elif defense_action == Action.RECOVER:
            hit_probability *= 1.12

        if attack_action == Action.HOOK and defense_action == Action.DODGE:
            hit_probability *= 1.08
        elif attack_action == Action.JAB and defense_action == Action.DODGE:
            hit_probability *= 0.92

        if self.rng.random() >= _clamp(hit_probability, 0.08, 0.95):
            return False, 0.0

        # Recent boxing data found roughly 4-10% fatigue-related punch-force loss.
        fatigue_power = 0.90 + 0.10 * stamina_ratio
        variance = self.rng.uniform(0.88, 1.12)
        defense_absorption = 1.0 / (0.85 + 0.15 * defender.profile.defense)
        damage = (
            spec.base_damage
            * attacker.profile.power
            * fatigue_power
            * variance
            * damage_multiplier
            * defense_absorption
        )
        return True, damage

    def _score_hit(self, fighter_index: int, hit: bool, damage: float) -> float:
        if not hit:
            return 0.0
        fighter = self.fighters[fighter_index]
        fighter.punches_landed += 1
        points = 1.0 + min(damage / 12.0, 1.0)
        fighter.score += points
        return points

    def _outcome(self) -> tuple[int | None, str]:
        knocked_out = [fighter.health <= 0.0 for fighter in self.fighters]
        if knocked_out == [True, True]:
            return None, "double KO draw"
        if knocked_out[0]:
            return 1, "KO"
        if knocked_out[1]:
            return 0, "KO"

        score_gap = self.fighters[0].score - self.fighters[1].score
        if abs(score_gap) > 0.5:
            return (0 if score_gap > 0 else 1), "decision"
        health_gap = self.fighters[0].health - self.fighters[1].health
        if abs(health_gap) > 1.0:
            return (0 if health_gap > 0 else 1), "decision"
        return None, "draw"

    def result(self) -> MatchResult:
        if not self.done:
            raise RuntimeError("The match is not complete")
        winner, method = self._outcome()
        return MatchResult(
            fighters=(copy.deepcopy(self.fighters[0]), copy.deepcopy(self.fighters[1])),
            winner=winner,
            method=method,
            exchanges=self.exchange,
            exchanges_per_round=self.exchanges_per_round,
            history=list(self.history),
        )


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_size: int = BoxingEnv.OBSERVATION_SIZE,
        hidden_size: int = 96,
    ) -> None:
        super().__init__()
        self.observation_size = observation_size
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, ACTION_COUNT)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observation: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(observation)
        return self.policy_head(features), self.value_head(features).squeeze(-1)

    def distribution(
        self, observation: Tensor, action_mask: Tensor
    ) -> tuple[Categorical, Tensor]:
        logits, value = self(observation)
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return Categorical(logits=masked_logits), value

    @torch.no_grad()
    def act(
        self,
        observation: Tensor,
        action_mask: Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        distribution, value = self.distribution(observation, action_mask)
        action = distribution.logits.argmax() if deterministic else distribution.sample()
        return (
            int(action.item()),
            float(distribution.log_prob(action).item()),
            float(value.item()),
        )


class PolicyAgent:
    def __init__(self, policy: ActorCritic, *, deterministic: bool = False) -> None:
        self.policy = policy
        self.deterministic = deterministic

    def choose_action(self, observation: Tensor, action_mask: Tensor) -> int:
        action, _, _ = self.policy.act(
            observation,
            action_mask,
            deterministic=self.deterministic,
        )
        return action


class HeuristicAgent:
    """A stochastic tactical baseline used for bootstrapping and evaluation."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, observation: Tensor, action_mask: Tensor) -> int:
        stamina = float(observation[1])
        last_opponent = observation[22:28]
        opponent_attacked = (
            float(last_opponent.sum()) > 0.0
            and int(last_opponent.argmax()) in {int(action) for action in ATTACK_ACTIONS}
        )

        if stamina < 0.20 and bool(action_mask[Action.RECOVER]) and self.rng.random() < 0.72:
            return int(Action.RECOVER)
        if opponent_attacked and self.rng.random() < 0.36:
            defensive = [
                action
                for action in (Action.GUARD, Action.DODGE)
                if bool(action_mask[action])
            ]
            if defensive:
                return int(self.rng.choice(defensive))

        actions = list(Action)
        weights = [0.34, 0.28, 0.18, 0.08, 0.06, 0.06]
        valid_actions = [
            action for action in actions if bool(action_mask[int(action)])
        ]
        valid_weights = [weights[int(action)] for action in valid_actions]
        return int(self.rng.choices(valid_actions, weights=valid_weights, k=1)[0])


class RandomAgent:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def choose_action(self, observation: Tensor, action_mask: Tensor) -> int:
        del observation
        valid = [index for index, allowed in enumerate(action_mask) if bool(allowed)]
        return self.rng.choice(valid)


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 320
    batch_episodes: int = 16
    ppo_epochs: int = 4
    mini_batch_size: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    entropy_coefficient: float = 0.015
    value_coefficient: float = 0.5
    max_gradient_norm: float = 0.5
    snapshot_interval: int = 64
    opponent_pool_size: int = 6
    max_exchanges: int = 60


@dataclass
class _EpisodeSamples:
    observations: list[Tensor]
    masks: list[Tensor]
    actions: list[int]
    log_probabilities: list[float]
    values: list[float]
    rewards: list[float]
    advantages: list[float]
    returns: list[float]


@dataclass(frozen=True)
class Evaluation:
    wins: int
    losses: int
    draws: int

    @property
    def score_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return (self.wins + 0.5 * self.draws) / total if total else 0.0


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def sample_profile(name: str, rng: random.Random) -> FighterProfile:
    return FighterProfile(
        name=name,
        power=rng.uniform(0.90, 1.10),
        speed=rng.uniform(0.90, 1.10),
        defense=rng.uniform(0.90, 1.10),
        conditioning=rng.uniform(0.90, 1.10),
    )


def simulate_match(
    first_agent: Agent,
    second_agent: Agent,
    *,
    profiles: tuple[FighterProfile, FighterProfile] | None = None,
    seed: int = 0,
    max_exchanges: int = 60,
) -> MatchResult:
    environment = BoxingEnv(
        profiles,
        seed=seed,
        max_exchanges=max_exchanges,
    )
    while not environment.done:
        first_action = first_agent.choose_action(
            environment.observation(0),
            environment.valid_action_mask(0),
        )
        second_action = second_agent.choose_action(
            environment.observation(1),
            environment.valid_action_mask(1),
        )
        environment.step(first_action, second_action)
    return environment.result()


def train_self_play(
    policy: ActorCritic,
    *,
    config: TrainingConfig | None = None,
    seed: int = 0,
    progress_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> list[dict[str, float]]:
    config = config or TrainingConfig()
    if config.episodes <= 0:
        return []

    set_global_seed(seed)
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    snapshots = [_snapshot(policy)]
    history: list[dict[str, float]] = []
    episode_batch: list[_EpisodeSamples] = []
    batch_returns: list[float] = []

    for episode in range(1, config.episodes + 1):
        if rng.random() < 0.35:
            opponent: Agent = HeuristicAgent(seed=rng.randrange(2**31))
        else:
            opponent_policy = ActorCritic()
            opponent_policy.load_state_dict(rng.choice(snapshots))
            opponent_policy.eval()
            opponent = PolicyAgent(opponent_policy)

        samples = _collect_episode(policy, opponent, config, rng)
        episode_batch.append(samples)
        batch_returns.append(sum(samples.rewards))

        should_update = (
            len(episode_batch) >= config.batch_episodes or episode == config.episodes
        )
        if should_update:
            metrics = _ppo_update(policy, optimizer, episode_batch, config)
            metrics["episode"] = float(episode)
            metrics["mean_return"] = sum(batch_returns) / len(batch_returns)
            history.append(metrics)
            episode_batch.clear()
            batch_returns.clear()
            if progress_callback is not None:
                progress_callback(episode, metrics)

        if episode % config.snapshot_interval == 0:
            snapshots.append(_snapshot(policy))
            snapshots = snapshots[-config.opponent_pool_size :]

    return history


def _collect_episode(
    policy: ActorCritic,
    opponent: Agent,
    config: TrainingConfig,
    rng: random.Random,
) -> _EpisodeSamples:
    learner_side = rng.randrange(2)
    profiles = (
        sample_profile("Fighter 1", rng),
        sample_profile("Fighter 2", rng),
    )
    environment = BoxingEnv(
        profiles,
        seed=rng.randrange(2**31),
        max_exchanges=config.max_exchanges,
    )
    observations: list[Tensor] = []
    masks: list[Tensor] = []
    actions: list[int] = []
    log_probabilities: list[float] = []
    values: list[float] = []
    rewards: list[float] = []

    while not environment.done:
        observation = environment.observation(learner_side)
        mask = environment.valid_action_mask(learner_side)
        action, log_probability, value = policy.act(observation, mask)
        opponent_side = 1 - learner_side
        opponent_action = opponent.choose_action(
            environment.observation(opponent_side),
            environment.valid_action_mask(opponent_side),
        )
        if learner_side == 0:
            first_reward, _, _ = environment.step(action, opponent_action)
            reward = first_reward
        else:
            _, second_reward, _ = environment.step(opponent_action, action)
            reward = second_reward

        observations.append(observation)
        masks.append(mask)
        actions.append(action)
        log_probabilities.append(log_probability)
        values.append(value)
        rewards.append(reward)

    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    generalized_advantage = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else 0.0
        delta = rewards[index] + config.gamma * next_value - values[index]
        generalized_advantage = (
            delta
            + config.gamma * config.gae_lambda * generalized_advantage
        )
        advantages[index] = generalized_advantage
        returns[index] = generalized_advantage + values[index]

    return _EpisodeSamples(
        observations=observations,
        masks=masks,
        actions=actions,
        log_probabilities=log_probabilities,
        values=values,
        rewards=rewards,
        advantages=advantages,
        returns=returns,
    )


def _ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    episodes: Sequence[_EpisodeSamples],
    config: TrainingConfig,
) -> dict[str, float]:
    observations = torch.stack(
        [item for episode in episodes for item in episode.observations]
    )
    masks = torch.stack([item for episode in episodes for item in episode.masks])
    actions = torch.tensor(
        [item for episode in episodes for item in episode.actions],
        dtype=torch.long,
    )
    old_log_probabilities = torch.tensor(
        [item for episode in episodes for item in episode.log_probabilities],
        dtype=torch.float32,
    )
    advantages = torch.tensor(
        [item for episode in episodes for item in episode.advantages],
        dtype=torch.float32,
    )
    returns = torch.tensor(
        [item for episode in episodes for item in episode.returns],
        dtype=torch.float32,
    )
    advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )

    metric_totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    update_count = 0
    sample_count = len(actions)

    for _ in range(config.ppo_epochs):
        permutation = torch.randperm(sample_count)
        for start in range(0, sample_count, config.mini_batch_size):
            indices = permutation[start : start + config.mini_batch_size]
            distribution, predicted_values = policy.distribution(
                observations[indices],
                masks[indices],
            )
            new_log_probabilities = distribution.log_prob(actions[indices])
            probability_ratio = (
                new_log_probabilities - old_log_probabilities[indices]
            ).exp()
            unclipped = probability_ratio * advantages[indices]
            clipped = probability_ratio.clamp(
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = (predicted_values - returns[indices]).pow(2).mean()
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), config.max_gradient_norm)
            optimizer.step()

            metric_totals["policy_loss"] += float(policy_loss.item())
            metric_totals["value_loss"] += float(value_loss.item())
            metric_totals["entropy"] += float(entropy.item())
            update_count += 1

    return {
        key: value / update_count
        for key, value in metric_totals.items()
    }


def evaluate_policy(
    policy: ActorCritic,
    *,
    episodes: int = 100,
    seed: int = 0,
    max_exchanges: int = 60,
) -> Evaluation:
    if episodes < 0:
        raise ValueError("episodes cannot be negative")
    wins = losses = draws = 0
    neural_agent = PolicyAgent(policy, deterministic=True)

    for episode in range(episodes):
        baseline = HeuristicAgent(seed=seed + episode)
        neural_side = episode % 2
        profiles = (
            FighterProfile("Fighter 1"),
            FighterProfile("Fighter 2"),
        )
        if neural_side == 0:
            result = simulate_match(
                neural_agent,
                baseline,
                profiles=profiles,
                seed=seed + 100_000 + episode,
                max_exchanges=max_exchanges,
            )
        else:
            result = simulate_match(
                baseline,
                neural_agent,
                profiles=profiles,
                seed=seed + 100_000 + episode,
                max_exchanges=max_exchanges,
            )

        if result.winner is None:
            draws += 1
        elif result.winner == neural_side:
            wins += 1
        else:
            losses += 1

    return Evaluation(wins=wins, losses=losses, draws=draws)


def save_policy(policy: ActorCritic, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "observation_size": policy.observation_size,
            "hidden_size": policy.hidden_size,
            "action_count": ACTION_COUNT,
            "state_dict": policy.state_dict(),
        },
        target,
    )


def load_policy(path: str | Path) -> ActorCritic:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("observation_size") != BoxingEnv.OBSERVATION_SIZE:
        raise ValueError("Model observation size is incompatible with this simulator")
    if checkpoint.get("action_count") != ACTION_COUNT:
        raise ValueError("Model action count is incompatible with this simulator")
    policy = ActorCritic(
        observation_size=checkpoint["observation_size"],
        hidden_size=checkpoint["hidden_size"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy.eval()
    return policy


def plot_match(result: MatchResult, path: str | Path) -> None:
    import matplotlib.pyplot as plt

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exchanges = [0] + [record.exchange for record in result.history]
    names = [fighter.profile.name for fighter in result.fighters]

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    series = (
        (
            [MAX_HEALTH] + [record.health[0] for record in result.history],
            [MAX_HEALTH] + [record.health[1] for record in result.history],
            "Health",
        ),
        (
            [MAX_STAMINA] + [record.stamina[0] for record in result.history],
            [MAX_STAMINA] + [record.stamina[1] for record in result.history],
            "Stamina",
        ),
        (
            [0.0] + [record.score[0] for record in result.history],
            [0.0] + [record.score[1] for record in result.history],
            "Effective-strike score",
        ),
    )
    for axis, (first_values, second_values, label) in zip(axes, series):
        axis.plot(exchanges, first_values, label=names[0], linewidth=2)
        axis.plot(exchanges, second_values, label=names[1], linewidth=2)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        for boundary in range(
            result.exchanges_per_round,
            result.exchanges,
            result.exchanges_per_round,
        ):
            axis.axvline(boundary, color="black", alpha=0.15, linestyle="--")

    axes[0].legend()
    axes[-1].set_xlabel("Exchange")
    figure.suptitle(result.summary())
    figure.tight_layout()
    figure.savefig(target, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _snapshot(policy: ActorCritic) -> dict[str, Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in policy.state_dict().items()
    }


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
