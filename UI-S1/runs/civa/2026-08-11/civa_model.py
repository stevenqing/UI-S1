from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class ProbabilityHead:
    def __init__(self, probability=None, model=None):
        if (probability is None) == (model is None):
            raise ValueError("CIVA probability head requires one implementation")
        self.probability = probability
        self.model = model

    def predict(self, features):
        if self.model is None:
            return np.full(len(features), self.probability, dtype=np.float64)
        return self.model.predict_proba(features)[:, 1]


@dataclass(frozen=True)
class UpliftModel:
    rescue_heads: tuple[ProbabilityHead, ...]
    harm_heads: tuple[ProbabilityHead, ...]

    def predict(self, features):
        rescue = np.column_stack([head.predict(features) for head in self.rescue_heads])
        harm = np.column_stack([head.predict(features) for head in self.harm_heads])
        return rescue - harm, rescue, harm


def _fit_head(features, target, weights, learner, seed):
    target = np.asarray(target, dtype=np.int8)
    weights = np.asarray(weights, dtype=np.float64)
    probability = float(np.average(target, weights=weights))
    if np.unique(target).size == 1:
        return ProbabilityHead(probability=probability)
    normalized_weights = weights * len(weights) / weights.sum()
    model = HistGradientBoostingClassifier(
        learning_rate=learner["learning_rate"],
        max_iter=learner["max_iter"],
        max_leaf_nodes=learner["max_leaf_nodes"],
        max_depth=learner["max_depth"],
        min_samples_leaf=learner["min_samples_leaf"],
        l2_regularization=learner["l2_regularization"],
        early_stopping=learner["early_stopping"],
        random_state=seed,
    )
    model.fit(features, target, sample_weight=normalized_weights)
    return ProbabilityHead(model=model)


def fit_uplift_model(features, delta, weights, learner, seed):
    delta = np.asarray(delta)
    if delta.ndim != 2 or delta.shape[0] != len(features):
        raise ValueError("CIVA uplift target shape mismatch")
    if not np.isin(delta, (-1, 0, 1)).all():
        raise ValueError("CIVA uplift target domain mismatch")
    rescue_heads = []
    harm_heads = []
    for expert in range(delta.shape[1]):
        rescue_heads.append(_fit_head(
            features, delta[:, expert] == 1, weights, learner, seed + expert * 10 + 1
        ))
        harm_heads.append(_fit_head(
            features, delta[:, expert] == -1, weights, learner, seed + expert * 10 + 2
        ))
    return UpliftModel(tuple(rescue_heads), tuple(harm_heads))