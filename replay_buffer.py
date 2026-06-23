from collections import deque
from typing import Any

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: deque[dict[str, Any]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: dict[str, Any]) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.choice(len(self.data), size=batch_size, replace=False)
        batch = [self.data[i] for i in idx]
        out: dict[str, torch.Tensor] = {}
        for key in batch[0].keys():
            values = [item[key] for item in batch]
            arr = np.stack(values)
            tensor = torch.as_tensor(arr, device=device)
            if arr.dtype == np.bool_:
                tensor = tensor.bool()
            elif np.issubdtype(arr.dtype, np.integer):
                tensor = tensor.long()
            else:
                tensor = tensor.float()
            out[key] = tensor
        return out

