from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from train import build_parser, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark 460-bit encoding candidates on short SZ-Tetris runs")
    parser.add_argument("--activation", type=str, default="dsilu")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output-root", type=str, default="results/encoding_bench")
    args = parser.parse_args()

    encodings = ["threshold460", "onehot460", "ordinal460"]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    summaries: list[tuple[str, float, int]] = []

    for encoding in encodings:
        train_args = build_parser().parse_args([])
        train_args.activation = args.activation
        train_args.encoding = encoding
        train_args.episodes = args.episodes
        train_args.log_every = args.log_every
        train_args.seed = args.seed
        train_args.lr = args.lr
        train_args.runs = 1
        train_args.output_dir = str(root / encoding)
        run_training(train_args)

        summary_path = root / encoding / "summary.json"
        import json

        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        summaries.append((encoding, float(summary["mean_final_average_score"]), int(summary["best_episode_scores"][0])))

    summaries.sort(key=lambda item: item[1], reverse=True)
    print("Encoding benchmark summary:")
    for encoding, final_avg, best_score in summaries:
        print(f"{encoding}: final_avg={final_avg:.3f}, best_score={best_score}")


if __name__ == "__main__":
    main()