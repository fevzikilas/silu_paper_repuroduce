# SILU Paper Reproduction

This repository is an experimental codebase focused on reproducing stochastic SZ-Tetris learning with shallow neural networks. The goal is to rebuild the first experimental setup from the paper as closely as possible, record learning curves, and make the results reproducible.

The focus of this project is not to build a general Tetris agent. The focus is specifically on the stochastic SZ-Tetris setting, where only `S` and `Z` pieces appear, and on approximating the paper's setup with a small network and TD($\lambda$)-based value learning.

## Paper Overview

This repository is based on the paper "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning" by Stefan Elfwing, Eiji Uchibe, and Kenji Doya.

Paper link:

- https://arxiv.org/abs/1702.03118
- https://arxiv.org/pdf/1702.03118

The paper has two main goals:

- to introduce the SiLU and dSiLU activation functions for reinforcement learning function approximation
- to show that on-policy reinforcement learning with eligibility traces and softmax action selection can be highly competitive without relying on the standard DQN recipe of experience replay plus target networks

The paper validates that claim across multiple settings rather than only one benchmark. It begins with shallow-network control experiments in Tetris-like domains and then extends the study to deeper reinforcement learning experiments in Atari 2600.

## Paper Goal and Main Idea

The paper is not only a benchmark report. Its main purpose is methodological.

At a high level, the paper argues that reinforcement learning with neural networks does not have to rely exclusively on the now-standard DQN-style recipe. Instead, it proposes a different combination of design choices:

- new activation functions tailored for value-function approximation in reinforcement learning
- on-policy temporal-difference learning with eligibility traces
- softmax-based action selection instead of a purely greedy policy with replay-driven stabilization

The central claim is that this combination can work extremely well in practice, even without experience replay and without a separate target network. In that sense, the paper is partly about activation functions and partly about a full learning recipe.

The authors introduce two related activation functions:

- `SiLU`, the sigmoid-weighted linear unit
- `dSiLU`, the derivative of SiLU

The motivation is that reinforcement learning function approximation has different optimization pressures than standard supervised learning. The paper studies whether these activations produce better behavior than more conventional hidden units when used inside value-based reinforcement learning agents.

## Paper Methods in Brief

Methodologically, the paper combines three main ingredients.

First, it uses neural networks as value-function approximators. In the shallow Tetris-style experiments, the network maps hand-designed binary features to scalar value estimates. In the deeper Atari experiments, convolutional and fully connected layers are used in a more standard deep reinforcement learning setting.

Second, it uses temporal-difference learning with eligibility traces. Instead of relying on replay buffers and delayed target copies, the learning signal is propagated online through TD($\lambda$) or Sarsa($\lambda$), depending on the experiment. This makes the method more directly on-policy and more tightly coupled to the current behavior policy.

Third, it uses softmax action selection with annealing. Rather than switching to hard argmax behavior too early, the policy gradually becomes more selective over time. This is important in the paper because the authors want exploration and value learning to remain compatible with online trace-based updates.

The overall methodology can be summarized as follows:

1. represent the environment state or afterstate with a neural-network input
2. estimate values with SiLU or dSiLU-based networks
3. select actions through softmax sampling
4. update online with TD($\lambda$) or Sarsa($\lambda$)
5. compare the resulting learning dynamics against established baselines

So the paper's experiments are not isolated case studies. They are the validation layer for a broader methodological proposal: activation design plus online reinforcement learning with traces.

## What Does the Paper Do?

This project aims to reconstruct a setup with the following characteristics:

- The board size is `10 x 20`.
- The environment contains only `S` and `Z` pieces.
- The agent evaluates possible placements at each step.
- The state or afterstate is represented as a fixed-length binary vector.
- That vector is scored by a shallow neural network with a small hidden layer.
- Learning is performed with TD($\lambda$) updates using eligibility traces.
- The policy is defined through softmax action selection over estimated values.

The purpose of this repository is to make that logic transparent in code: how the environment is defined, how features are extracted, what the network evaluates, how updates are performed, and how learning curves are reported.

## Experiments in the Paper

The paper is broader than the first SZ-Tetris experiment implemented in this repository. At a high level, it reports results for three groups of experiments:

1. Stochastic SZ-Tetris with shallow TD($\lambda$) agents.
2. Standard Tetris on a small `10 x 10` board, again with shallow agents.
3. Atari 2600 experiments with deep Sarsa($\lambda$) agents using SiLU and dSiLU hidden units.

That broader context matters because the paper is not only about solving SZ-Tetris. SZ-Tetris is the first controlled testbed used to show that the proposed activations and training setup work well even with relatively small architectures. The later Tetris and Atari experiments are intended to show that the same design ideas remain useful as the environments become more complex.

At the moment, this repository is focused on the first of those experiments: stochastic SZ-Tetris. The code structure was intentionally kept modular so that the remaining experiments can be added later without rewriting the entire training pipeline.

## How Is the Approach Implemented in This Codebase?

The current implementation is afterstate-based. In other words, the network does not directly output an action index. Instead, the environment generates all legal placements for the current piece, extracts a feature vector for the resulting board of each placement, lets the agent evaluate those candidate afterstates, and then samples one of them using softmax.

This approach was chosen because, in SZ-Tetris, what matters is the board left behind by the selected move. For that reason, the network output is not a vector of action logits, but a single scalar afterstate value.

The current training pipeline works as follows:

1. The environment generates all legal afterstates for the current piece.
2. Each afterstate is converted into a `460`-dimensional feature vector.
3. The shallow network produces a single value for that vector.
4. The agent samples an action using softmax over those values.
5. The environment returns a shaped reward, while episode score is tracked separately through cleared lines.
6. The network weights are updated with TD($\lambda$) and eligibility traces.

## Project Structure

- `environments/`
  Contains the SZ-Tetris environment and the base environment interface.
- `environments/sz_tetris.py`
  Contains piece placement, line clearing, hole counting, legal afterstate generation, and the 460-bit feature encodings.
- `models.py`
  Defines the shallow network architecture and the `ReLU`, `SiLU`, `dSiLU`, and `sigmoid` activations.
- `agent.py`
  Contains softmax action selection, temperature scheduling, TD($\lambda$) updates, and eligibility trace logic.
- `train.py`
  Main training entry point. Creates experiment directories and writes CSV and JSON summaries.
- `plot_results.py`
  Produces standard and paper-style learning curves from saved experiments.
- `benchmark_encodings.py`
  Compares different 460-bit feature representations on short runs.
- `results/`
  Stores training outputs, summaries, and generated figures.

## Environment and Feature Representation

The environment samples only `S` and `Z` pieces. For each legal placement, it computes the following:

- the resulting board
- the number of cleared lines
- the number of holes
- the shaped reward
- a 460-dimensional feature vector

The current shaped reward is defined as:

$$
r = e^{-\text{holes}/33}
$$

However, when comparing against the paper, the main metric of interest is episode score, meaning the total number of cleared lines. This distinction matters: the reward used for learning is not identical to the metric used for evaluation.

At the moment, the codebase includes three candidate 460-bit encodings:

- `threshold460`
- `onehot460`
- `ordinal460`

In short experiments so far, `threshold460` has produced the strongest early learning signal.

## Model

The model is a small shallow network:

- input size: `460`
- hidden layer: `50` units
- output: `1` scalar value

It is designed to estimate a single value for each afterstate. The network is intentionally kept small in order to stay close to the paper's shallow-network setup.

## Learning Algorithm

The agent learns with TD($\lambda$). Parameter gradients are computed with standard backpropagation, then eligibility trace tensors are maintained, and the weights are updated with the following logic:

- compute the current state value
- build the one-step TD target
- compute the TD error
- decay the traces by $\gamma \lambda$
- add the new gradient to the traces
- update the parameters in the direction of the TD error

Action selection is not greedy. Instead, afterstate values are sampled through softmax. The temperature is not annealed linearly; instead, inverse temperature increases over time so that the policy becomes more selective as training progresses.

## Setup

This project runs in Python. The main dependencies are:

- `numpy`
- `torch`
- `matplotlib`

This repository has been used with a local virtual environment. On Windows, the example commands use `.venv\Scripts\python.exe`.

## Running Training

Example short training run:

```powershell
.venv\Scripts\python.exe -u train.py --activation dsilu --encoding threshold460 --lr 0.001 --episodes 200 --log-every 50 --output-dir results\example_run
```

Example longer run:

```powershell
.venv\Scripts\python.exe -u train.py --activation dsilu --encoding threshold460 --lr 0.001 --episodes 2000 --log-every 1000 --output-dir results\pre_analysis_2k_threshold460
```

Main arguments:

- `--activation`: `relu`, `silu`, `dsilu`, `sigmoid`
- `--encoding`: `threshold460`, `onehot460`, `ordinal460`
- `--lr`: learning rate
- `--episodes`: number of episodes
- `--lambda_`: TD($\lambda$) parameter
- `--gamma`: discount factor
- `--tau-start`: initial temperature
- `--tau-k`: inverse-temperature growth factor
- `--log-every`: logging interval in episodes
- `--runs`: number of repeated runs
- `--seed`: random seed
- `--output-dir`: output directory

For each run, `train.py` writes:

- `run_XX.csv`
- `summary.json`

CSV files are now written not only at the end of a run, but also at each logging interval, so long experiments can be monitored while they are still running.

## Comparing Encodings

To compare 460-bit feature candidates on short runs:

```powershell
.venv\Scripts\python.exe -u benchmark_encodings.py --activation dsilu --episodes 200 --log-every 50 --output-root results\encoding_bench_200
```

This script trains each encoding separately and prints a ranked summary at the end.

## Plotting Results

Standard learning curve:

```powershell
.venv\Scripts\python.exe -u plot_results.py results\example_run --window 100 --save results\example_run\curve.png
```

Paper-style aggregated score plot:

```powershell
.venv\Scripts\python.exe -u plot_results.py results\example_run --mode paper --metric scores --aggregate-every 1000 --save results\example_run\figure2_like_score.png
```

## Output Format

Each `run_XX.csv` file contains the following columns:

- `episode`
- `score`
- `shaped_reward`
- `tau`

The `summary.json` file stores the experiment configuration and summary performance values.

## Current Status

This repository provides a working research scaffold for reproducing the paper's first SZ-Tetris experiment, but exact result matching has not yet been completed. In particular, the exact meaning of the `460`-dimensional binary representation still appears to be critical. For that reason, this project is organized not only to run training, but also to compare representation choices and observe which design behaves more like the paper.

In short benchmarks so far, `threshold460` appears stronger than the other tested encoding candidates, so it is currently the main candidate for longer runs.

The other experiments from the paper are not yet implemented here in full. In practical terms, that means this repository currently documents and reproduces the first shallow-network result, while the `10 x 10` Tetris setting and the Atari deep Sarsa($\lambda$) experiments remain future extensions.

## Why This README Exists

This is not a product repository. It is an experimental reproduction repository. Because of that, the purpose of this README is clarity rather than presentation:

- to explain which paper idea is being reconstructed
- to show which part of the code handles which responsibility
- to make the experiment workflow explicit
- to state honestly which parts are still under investigation

The most important question in this repository is not simply whether the agent learns. The more important question is whether it learns under a representation and training setup that is faithful enough to the paper.