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

    def extend(self, transitions: list[dict[str, Any]]) -> None:
        self.data.extend(transitions)

    def _stack(self, key: str) -> np.ndarray:
        return np.stack([item[key] for item in self.data])

    def tensors(self, gamma: float, gae_lambda: float, device: torch.device) -> dict[str, torch.Tensor]:
        values = self._stack("values").astype(np.float32)
        if self.data and "next_values" in self.data[0]:
            next_values_arr = self._stack("next_values").astype(np.float32)
        else:
            next_values_arr = np.zeros_like(values, dtype=np.float32)
            if len(values) > 1:
                next_values_arr[:-1] = values[1:]
        rewards = self._stack("reward").astype(np.float32).reshape(-1)
        dones = self._stack("done").astype(np.float32).reshape(-1)
        t, n = values.shape
        advantages = np.zeros((t, n), dtype=np.float32)
        last_gae = np.zeros(n, dtype=np.float32)
        for step in range(t - 1, -1, -1):
            next_nonterminal = 1.0 - dones[step]
            delta = rewards[step] + gamma * next_values_arr[step] * next_nonterminal - values[step]
            last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
            advantages[step] = last_gae
        returns = advantages + values

        keys = [
            "global_node_inputs",
            "global_edge_mask",
            "global_node_padding_mask",
            "current_node_indices",
            "candidate_node_indices",
            "candidate_padding_mask",
            "action_mask",
            "node_inputs",
            "node_padding_mask",
            "uav_state",
            "prev_option",
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
        out["next_values"] = torch.as_tensor(next_values_arr, device=device).float()
        out["rewards"] = torch.as_tensor(rewards, device=device).float()
        return out

    @staticmethod
    def minibatches(
        tensors: dict[str, torch.Tensor],
        minibatch_size: int,
        num_minibatches: int = 0,
    ) -> Iterator[dict[str, torch.Tensor]]:
        size = tensors["actions"].shape[0]
        indices = np.arange(size)
        np.random.shuffle(indices)
        if num_minibatches and num_minibatches > 0:
            sampler = [chunk for chunk in np.array_split(indices, num_minibatches) if len(chunk) > 0]
        else:
            step = max(1, min(minibatch_size, size))
            sampler = [indices[start : start + step] for start in range(0, size, step)]
        for batch_indices in sampler:
            idx = torch.as_tensor(batch_indices, device=tensors["actions"].device).long()
            yield {key: value.index_select(0, idx) for key, value in tensors.items()}
