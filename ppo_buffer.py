from typing import Any, Iterator

import numpy as np
import torch


class PPORolloutBuffer:
    def __init__(self):
        self.data: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self.data)

    def clear(self) -> None:
        self.data.clear()

    def add(self, transition: dict[str, Any]) -> None:
        self.data.append(transition)

    def _stack(self, key: str) -> np.ndarray:
        return np.stack([item[key] for item in self.data])

    def tensors(self, gamma: float, gae_lambda: float, device: torch.device) -> dict[str, torch.Tensor]:
        values = self._stack("values").astype(np.float32)
        rewards = self._stack("reward").astype(np.float32).reshape(-1)
        dones = self._stack("done").astype(np.float32).reshape(-1)
        t, n = values.shape
        advantages = np.zeros((t, n), dtype=np.float32)
        last_gae = np.zeros(n, dtype=np.float32)
        for step in range(t - 1, -1, -1):
            if step == t - 1:
                next_values = np.zeros(n, dtype=np.float32)
            else:
                next_values = values[step + 1]
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values

        keys = [
            "node_inputs",
            "node_padding_mask",
            "action_mask",
            "uav_state",
            "prev_option",
            "team_summary",
            "actions",
            "terminations",
            "log_probs",
        ]
        out: dict[str, torch.Tensor] = {}
        for key in keys:
            arr = self._stack(key)
            tensor = torch.as_tensor(arr, device=device)
            if arr.dtype == np.bool_:
                tensor = tensor.bool()
            elif np.issubdtype(arr.dtype, np.integer):
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            out[key] = tensor
        out["advantages"] = torch.as_tensor(advantages, device=device).float()
        out["returns"] = torch.as_tensor(returns, device=device).float()
        out["values"] = torch.as_tensor(values, device=device).float()
        out["rewards"] = torch.as_tensor(rewards, device=device).float()
        return out

    @staticmethod
    def minibatches(tensors: dict[str, torch.Tensor], minibatch_size: int) -> Iterator[dict[str, torch.Tensor]]:
        size = tensors["actions"].shape[0]
        indices = np.arange(size)
        np.random.shuffle(indices)
        step = max(1, min(minibatch_size, size))
        for start in range(0, size, step):
            idx = torch.as_tensor(indices[start : start + step], device=tensors["actions"].device).long()
            yield {key: value.index_select(0, idx) for key, value in tensors.items()}
