from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_run_csv(csv_path: Path) -> dict[str, np.ndarray]:
    episodes = []
    scores = []
    shaped_rewards = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episodes.append(int(row["episode"]))
            scores.append(float(row["score"]))
            shaped_rewards.append(float(row["shaped_reward"]))

    return {
        "episodes": np.array(episodes, dtype=np.int32),
        "scores": np.array(scores, dtype=np.float32),
        "shaped_rewards": np.array(shaped_rewards, dtype=np.float32),
    }


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="valid")


def chunked_means(values: np.ndarray, chunk_size: int) -> np.ndarray:
    if chunk_size <= 1:
        return values.copy()
    trimmed = len(values) // chunk_size * chunk_size
    if trimmed == 0:
        return np.array([], dtype=np.float32)
    return values[:trimmed].reshape(-1, chunk_size).mean(axis=1)


def plot_metric(ax, episodes: np.ndarray, values: np.ndarray, label: str, window: int, alpha: float) -> None:
    ax.plot(episodes, values, alpha=alpha, linewidth=1.0, label=f"{label} raw")
    smooth = moving_average(values, window)
    if len(smooth) == len(values):
        smooth_episodes = episodes
    else:
        smooth_episodes = episodes[window - 1 :]
    ax.plot(smooth_episodes, smooth, linewidth=2.0, label=f"{label} ma{window}")


def plot_paper_style(result_dir: Path, metric: str, aggregate_every: int, save_path: Path | None) -> None:
    run_files = sorted(result_dir.glob("run_*.csv"))
    if not run_files:
        raise FileNotFoundError(f"No run_*.csv files found in {result_dir}")

    color_map = {
        "relu": "blue",
        "silu": "magenta",
        "dsilu": "red",
        "sigmoid": "black",
    }
    activation = result_dir.name.lower()
    chosen_color = next((value for key, value in color_map.items() if key in activation), "tab:blue")

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    aggregated_runs = []
    x_axis = None

    for run_file in run_files:
        run_name = run_file.stem
        run_data = load_run_csv(run_file)
        series = run_data[metric]
        aggregated = chunked_means(series, aggregate_every)
        if aggregated.size == 0:
            continue
        x_axis = np.arange(1, len(aggregated) + 1, dtype=np.int32)
        aggregated_runs.append(aggregated)
        ax.plot(x_axis, aggregated, linestyle="--", linewidth=1.0, color=chosen_color, alpha=0.65, label=run_name)

    if not aggregated_runs or x_axis is None:
        raise ValueError(f"No aggregated data could be produced from {result_dir}")

    aggregated_matrix = np.vstack(aggregated_runs)
    mean_curve = aggregated_matrix.mean(axis=0)
    ax.plot(x_axis, mean_curve, linestyle="-", linewidth=3.0, color=chosen_color, label="mean")

    ax.set_xlabel(f"Episodes ({aggregate_every:,})")
    ax.set_ylabel("Mean score" if metric == "scores" else "Mean shaped reward")
    ax.set_title(f"SZ-Tetris {metric} learning curve\n{result_dir.name}")
    ax.grid(alpha=0.3)
    ax.set_xlim(left=1)
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 6:
        handles = [handles[-1]]
        labels = [labels[-1]]
    ax.legend(handles, labels, loc="lower right")
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def plot_directory(result_dir: Path, window: int, save_path: Path | None) -> None:
    run_files = sorted(result_dir.glob("run_*.csv"))
    if not run_files:
        raise FileNotFoundError(f"No run_*.csv files found in {result_dir}")

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(f"SZ-Tetris Learning Curves\n{result_dir}")

    score_runs = []
    reward_runs = []
    base_episodes = None

    for run_file in run_files:
        run_name = run_file.stem
        run_data = load_run_csv(run_file)
        base_episodes = run_data["episodes"]
        score_runs.append(run_data["scores"])
        reward_runs.append(run_data["shaped_rewards"])
        plot_metric(axes[0], run_data["episodes"], run_data["scores"], run_name, window, alpha=0.18)
        plot_metric(axes[1], run_data["episodes"], run_data["shaped_rewards"], run_name, window, alpha=0.18)

    if len(score_runs) > 1 and base_episodes is not None:
        score_matrix = np.vstack(score_runs)
        reward_matrix = np.vstack(reward_runs)
        axes[0].plot(base_episodes, score_matrix.mean(axis=0), color="black", linewidth=2.5, label="mean raw")
        axes[1].plot(base_episodes, reward_matrix.mean(axis=0), color="black", linewidth=2.5, label="mean raw")

        mean_score_smooth = moving_average(score_matrix.mean(axis=0), window)
        mean_reward_smooth = moving_average(reward_matrix.mean(axis=0), window)
        smooth_episodes = base_episodes if len(mean_score_smooth) == len(base_episodes) else base_episodes[window - 1 :]
        axes[0].plot(smooth_episodes, mean_score_smooth, color="red", linewidth=3.0, label=f"mean ma{window}")
        axes[1].plot(smooth_episodes, mean_reward_smooth, color="red", linewidth=3.0, label=f"mean ma{window}")

    axes[0].set_ylabel("Episode Score")
    axes[1].set_ylabel("Shaped Reward")
    axes[1].set_xlabel("Episode")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=8, ncol=2)
    axes[1].legend(loc="upper left", fontsize=8, ncol=2)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        print(f"Saved plot to: {save_path}")
    else:
        plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot saved SZ-Tetris training curves")
    parser.add_argument("result_dir", type=str, help="Directory containing run_*.csv files")
    parser.add_argument("--window", type=int, default=1000, help="Moving average window")
    parser.add_argument("--save", type=str, default=None, help="Optional output image path")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "paper"], help="Plot mode")
    parser.add_argument("--metric", type=str, default="scores", choices=["scores", "shaped_rewards"], help="Metric to plot in paper mode")
    parser.add_argument("--aggregate-every", type=int, default=1000, help="Episodes per point in paper mode")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    save_path = Path(args.save) if args.save else None
    if args.mode == "paper":
        plot_paper_style(Path(args.result_dir), args.metric, args.aggregate_every, save_path)
    else:
        plot_directory(Path(args.result_dir), args.window, save_path)