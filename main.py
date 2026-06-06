from __future__ import annotations

import argparse
from pathlib import Path

from neuroboxing import (
    ActorCritic,
    FighterProfile,
    HeuristicAgent,
    PolicyAgent,
    TrainingConfig,
    evaluate_policy,
    load_policy,
    plot_match,
    save_policy,
    set_global_seed,
    simulate_match,
    train_self_play,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a neural boxing policy with masked PPO self-play."
    )
    parser.add_argument("--episodes", type=int, default=320, help="Self-play training episodes.")
    parser.add_argument(
        "--eval-matches", type=int, default=100, help="Matches against the tactical baseline."
    )
    parser.add_argument("--seed", type=int, default=7, help="Reproducibility seed.")
    parser.add_argument(
        "--max-exchanges", type=int, default=60, help="Maximum exchanges in a match."
    )
    parser.add_argument(
        "--plot", type=Path, default=Path("boxing_match.png"), help="Showcase plot path."
    )
    parser.add_argument(
        "--model-out", type=Path, default=Path("neuroboxer.pt"), help="Trained model path."
    )
    parser.add_argument(
        "--load-model", type=Path, help="Load a saved policy instead of training a new one."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if args.load_model:
        policy = load_policy(args.load_model)
        print(f"Loaded policy from {args.load_model}")
    else:
        policy = ActorCritic()
        config = TrainingConfig(
            episodes=args.episodes,
            max_exchanges=args.max_exchanges,
        )

        def report(episode: int, metrics: dict[str, float]) -> None:
            print(
                f"Episode {episode:>4}/{config.episodes}: "
                f"return={metrics['mean_return']:+.3f} "
                f"policy_loss={metrics['policy_loss']:+.3f} "
                f"entropy={metrics['entropy']:.3f}"
            )

        print(
            f"Training {sum(parameter.numel() for parameter in policy.parameters()):,} "
            "parameters with masked PPO self-play..."
        )
        train_self_play(policy, config=config, seed=args.seed, progress_callback=report)
        save_policy(policy, args.model_out)
        print(f"Saved policy to {args.model_out}")

    evaluation = evaluate_policy(
        policy,
        episodes=args.eval_matches,
        seed=args.seed + 10_000,
        max_exchanges=args.max_exchanges,
    )
    print(
        "Evaluation vs tactical baseline: "
        f"{evaluation.wins} wins, {evaluation.draws} draws, "
        f"{evaluation.losses} losses ({evaluation.score_rate:.1%} score rate)"
    )

    showcase = simulate_match(
        PolicyAgent(policy, deterministic=True),
        HeuristicAgent(seed=args.seed + 20_000),
        profiles=(
            FighterProfile("NeuroBoxer"),
            FighterProfile("Tactical Baseline"),
        ),
        seed=args.seed + 30_000,
        max_exchanges=args.max_exchanges,
    )
    print(showcase.summary())
    plot_match(showcase, args.plot)
    print(f"Saved showcase plot to {args.plot}")


if __name__ == "__main__":
    main()
