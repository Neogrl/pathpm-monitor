import numpy as np

from config import Config
from nodes import NodeBuilder


class RandomBaseline:
    def select(self, cfg: Config, batch, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros(cfg.n_uavs, dtype=np.int64)
        for i in range(cfg.n_uavs):
            valid = np.flatnonzero(~batch.action_mask[i] & ~batch.node_padding_mask[i])
            actions[i] = int(rng.choice(valid)) if len(valid) else 0
        return actions


class HeuristicBaseline:
    def select(self, cfg: Config, batch, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros(cfg.n_uavs, dtype=np.int64)
        for i in range(cfg.n_uavs):
            valid = np.flatnonzero(~batch.action_mask[i] & ~batch.node_padding_mask[i])
            if len(valid) == 0:
                actions[i] = 0
                continue
            features = batch.node_inputs[i, valid]
            score = 1.3 * features[:, 5] + 1.0 * features[:, 8] + 1.1 * features[:, 11] - 0.6 * features[:, 10] - 0.05 * features[:, 2]
            actions[i] = int(valid[np.argmax(score)])
        return actions


class CoverageBaseline:
    def select(self, cfg: Config, batch, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros(cfg.n_uavs, dtype=np.int64)
        for i in range(cfg.n_uavs):
            valid = np.flatnonzero(~batch.action_mask[i] & ~batch.node_padding_mask[i])
            if len(valid) == 0:
                continue
            features = batch.node_inputs[i, valid]
            score = features[:, 9] - 0.5 * features[:, 10] - 0.03 * features[:, 2]
            actions[i] = int(valid[np.argmax(score)])
        return actions


class SearchGreedyBaseline:
    def select(self, cfg: Config, batch, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros(cfg.n_uavs, dtype=np.int64)
        for i in range(cfg.n_uavs):
            valid = np.flatnonzero(~batch.action_mask[i] & ~batch.node_padding_mask[i])
            if len(valid) == 0:
                continue
            features = batch.node_inputs[i, valid]
            score = features[:, 8] + 0.5 * features[:, 13] - 0.4 * features[:, 10] - 0.03 * features[:, 2]
            actions[i] = int(valid[np.argmax(score)])
        return actions


class PHDGreedyBaseline:
    def select(self, cfg: Config, batch, rng: np.random.Generator) -> np.ndarray:
        actions = np.zeros(cfg.n_uavs, dtype=np.int64)
        for i in range(cfg.n_uavs):
            valid = np.flatnonzero(~batch.action_mask[i] & ~batch.node_padding_mask[i])
            if len(valid) == 0:
                continue
            features = batch.node_inputs[i, valid]
            score = features[:, 5] + 0.5 * features[:, 12] - 0.4 * features[:, 10] - 0.03 * features[:, 2]
            actions[i] = int(valid[np.argmax(score)])
        return actions
