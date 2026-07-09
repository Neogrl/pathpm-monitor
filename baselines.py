import numpy as np

from config import Config
from nodes import NODE_INPUT_INDEX


COVERAGE_AGE_VALUE = NODE_INPUT_INDEX["coverage_age_value"]
OVERLAP = NODE_INPUT_INDEX["overlap"]
DISTANCE = NODE_INPUT_INDEX["candidate_distance_norm"]


def coverage_score(features: np.ndarray) -> np.ndarray:
    return features[:, COVERAGE_AGE_VALUE] - 0.5 * features[:, OVERLAP] - 0.05 * features[:, DISTANCE]


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
            score = coverage_score(features)
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
            score = coverage_score(features)
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
            score = coverage_score(features)
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
            score = coverage_score(features)
            actions[i] = int(valid[np.argmax(score)])
        return actions
