from __future__ import annotations

import argparse
import csv
import json
from statistics import mean
from datetime import datetime
from pathlib import Path

import numpy as np

from agent import TDLambdaAgent
from environments.sz_tetris import SZTetris
from models import ShallowNetwork


def prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"sz_tetris_{args.activation}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_run_csv(output_dir: Path, run_index: int, rows: list[dict[str, float | int]]) -> None:
    csv_path = output_dir / f"run_{run_index:02d}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "score",
                "shaped_reward",
                "tau",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_run_csv(output_dir: Path, run_index: int, rows: list[dict[str, float | int]]) -> None:
    if not rows:
        return

    csv_path = output_dir / f"run_{run_index:02d}.csv"
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode",
                "score",
                "shaped_reward",
                "tau",
            ],
        )
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def write_summary(output_dir: Path, args: argparse.Namespace, final_score_windows: list[float], best_scores: list[int]) -> None:
    summary = {
        "activation": args.activation,
        "encoding": args.encoding,
        "lr": args.lr,
        "episodes": args.episodes,
        "lambda": args.lambda_,
        "gamma": args.gamma,
        "tau_start": args.tau_start,
        "tau_k": args.tau_k,
        "log_every": args.log_every,
        "runs": args.runs,
        "seed": args.seed,
        "final_average_scores": final_score_windows,
        "best_episode_scores": best_scores,
        "mean_final_average_score": mean(final_score_windows) if final_score_windows else 0.0,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def run_training(args: argparse.Namespace) -> None:
    output_dir = prepare_output_dir(args)
    final_score_windows = []
    best_scores = []

    for run in range(args.runs):
        run_csv_path = output_dir / f"run_{run + 1:02d}.csv"
        if run_csv_path.exists():
            run_csv_path.unlink()

        seed = None if args.seed is None else args.seed + run
        env = SZTetris(seed=seed, encoding=args.encoding)
        model = ShallowNetwork(activation=args.activation)
        agent = TDLambdaAgent(
            model=model,
            lambda_=args.lambda_,
            gamma=args.gamma,
            alpha=args.lr,
            tau_start=args.tau_start,
            tau_k=args.tau_k,
        )

        episode_scores = []
        episode_rewards = []
        run_rows = []
        pending_rows = []

        for episode in range(1, args.episodes + 1):
            env.reset()
            agent.reset_traces()
            shaped_reward_total = 0.0
            episode_score = 0

            legal_afterstates = env.get_legal_afterstates()
            if not legal_afterstates:
                episode_scores.append(0)
                episode_rewards.append(0.0)
                continue

            selection = agent.select_action(np.stack([item.features for item in legal_afterstates]))
            current_afterstate = legal_afterstates[selection.index]
            done = False

            while not done:
                _, reward, done, info = env.step(current_afterstate.action)
                shaped_reward_total += reward
                episode_score += int(info["lines_cleared"])

                if done:
                    agent.update(current_afterstate.features, reward, None, True)
                    break

                next_afterstates = env.get_legal_afterstates()
                if not next_afterstates:
                    agent.update(current_afterstate.features, reward, None, True)
                    break

                next_selection = agent.select_action(np.stack([item.features for item in next_afterstates]))
                next_afterstate = next_afterstates[next_selection.index]
                agent.update(current_afterstate.features, reward, next_afterstate.features, False)
                current_afterstate = next_afterstate

            episode_scores.append(episode_score)
            episode_rewards.append(shaped_reward_total)
            run_rows.append(
                {
                    "episode": episode,
                    "score": episode_score,
                    "shaped_reward": round(shaped_reward_total, 6),
                    "tau": round(agent.current_tau(), 6),
                }
            )
            pending_rows.append(run_rows[-1])

            if episode % args.log_every == 0:
                score_window = episode_scores[-args.log_every :]
                reward_window = episode_rewards[-args.log_every :]
                append_run_csv(output_dir, run + 1, pending_rows)
                pending_rows.clear()
                print(
                    f"Run {run + 1} Episode {episode}, "
                    f"Avg Score ({args.log_every}): {mean(score_window):.3f}, "
                    f"Avg Shaped Reward ({args.log_every}): {mean(reward_window):.3f}, "
                    f"Tau: {agent.current_tau():.6f}, Beta: {agent.beta():.3f}"
                )

        final_window = episode_scores[-args.log_every :] if len(episode_scores) >= args.log_every else episode_scores
        final_avg = mean(final_window) if final_window else 0.0
        final_score_windows.append(final_avg)
        best_scores.append(max(episode_scores) if episode_scores else 0)
        append_run_csv(output_dir, run + 1, pending_rows)
        print(
            f"Run {run + 1} complete. Final Avg Score ({len(final_window)}): {final_avg:.3f}, "
            f"Best Episode Score: {max(episode_scores) if episode_scores else 0}"
        )

    if len(final_score_windows) > 1:
        print(
            f"Across {args.runs} runs, Mean Final Avg Score: {mean(final_score_windows):.3f}, "
            f"Best Final Avg Score: {max(final_score_windows):.3f}"
        )

    write_summary(output_dir, args, final_score_windows, best_scores)
    print(f"Saved results to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a shallow TD(lambda) agent on stochastic SZ-Tetris")
    parser.add_argument("--activation", type=str, default="dsilu", choices=["relu", "silu", "dsilu", "sigmoid"])
    parser.add_argument("--encoding", type=str, default="threshold460", choices=["threshold460", "onehot460", "ordinal460"])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--episodes", type=int, default=200_000)
    parser.add_argument("--lambda_", type=float, default=0.55)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau-start", type=float, default=0.5)
    parser.add_argument("--tau-k", type=float, default=0.00025)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser


if __name__ == "__main__":
    run_training(build_parser().parse_args())