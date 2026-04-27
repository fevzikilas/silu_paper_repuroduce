from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ActionSelection:
    index: int
    probabilities: np.ndarray
    values: np.ndarray


class TDLambdaAgent:
    def __init__(
        self,
        model: torch.nn.Module,
        lambda_: float = 0.55,
        gamma: float = 0.99,
        alpha: float = 0.001,
        tau_start: float = 0.5,
        tau_k: float = 0.00025,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.lambda_ = lambda_
        self.gamma = gamma
        self.alpha = alpha
        self.tau_start = tau_start
        self.tau_k = tau_k
        self.device = device
        self.action_steps = 0
        self.eligibility_traces = [torch.zeros_like(parameter, device=device) for parameter in self.model.parameters()]

    def reset_traces(self) -> None:
        for trace in self.eligibility_traces:
            trace.zero_()

    def beta(self) -> float:
        return (1.0 / max(self.tau_start, 1e-6)) + self.tau_k * self.action_steps

    def current_tau(self) -> float:
        return 1.0 / self.beta()

    def select_action(self, action_features: np.ndarray) -> ActionSelection:
        feature_tensor = torch.as_tensor(action_features, dtype=torch.float32, device=self.device)
        beta = self.beta()
        with torch.no_grad():
            values = self.model(feature_tensor)
            probabilities = torch.softmax(values * beta, dim=0)
        index = int(torch.multinomial(probabilities, 1).item())
        self.action_steps += 1
        return ActionSelection(
            index=index,
            probabilities=probabilities.detach().cpu().numpy(),
            values=values.detach().cpu().numpy(),
        )

    def evaluate(self, feature_vector: np.ndarray) -> torch.Tensor:
        feature_tensor = torch.as_tensor(feature_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.model(feature_tensor).squeeze(0)

    def update(self, current_features: np.ndarray, reward: float, next_features: np.ndarray | None, done: bool) -> float:
        value = self.evaluate(current_features)
        with torch.no_grad():
            next_value = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            if not done and next_features is not None:
                next_value = self.evaluate(next_features)
            target = torch.tensor(reward, dtype=torch.float32, device=self.device)
            if not done:
                target = target + self.gamma * next_value

        delta = target - value
        self.model.zero_grad()
        value.backward()

        with torch.no_grad():
            for trace, parameter in zip(self.eligibility_traces, self.model.parameters()):
                if parameter.grad is None:
                    continue
                trace.mul_(self.gamma * self.lambda_)
                trace.add_(parameter.grad)
                parameter.add_(self.alpha * delta * trace)

        return float(delta.detach().cpu().item())