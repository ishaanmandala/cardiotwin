"""Small, dependency-free survival analysis: Cox model, C-index, Kaplan-Meier.

Used for the "future heart" experiment on MUSIC, where we ask whether the heart
CardioTwin reads from today's ECG predicts heart-failure events over the next
four years.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class CoxModel:
    features: list[str]
    coef: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    base_times: np.ndarray
    base_cumhaz: np.ndarray

    def linear_predictor(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, float) - self.mean) / self.scale
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return Z @ self.coef

    def risk_at(self, X: np.ndarray, t: float) -> np.ndarray:
        """Probability of an event by time ``t`` (Breslow baseline)."""
        idx = np.searchsorted(self.base_times, t, side="right") - 1
        h0 = self.base_cumhaz[idx] if idx >= 0 else 0.0
        return 1.0 - np.exp(-h0 * np.exp(self.linear_predictor(X)))

    def hazard_ratios(self) -> dict[str, float]:
        """Hazard ratio per 1 SD of each feature."""
        return {f: float(np.exp(c)) for f, c in zip(self.features, self.coef)}

    def to_dict(self) -> dict:
        return {"features": self.features, "coef": self.coef.tolist(), "mean": self.mean.tolist(),
                "scale": self.scale.tolist(), "base_times": self.base_times.tolist(),
                "base_cumhaz": self.base_cumhaz.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> CoxModel:
        return cls(list(d["features"]), *(np.asarray(d[k], float) for k in
                                          ("coef", "mean", "scale", "base_times", "base_cumhaz")))


def fit_cox(X: np.ndarray, time: np.ndarray, event: np.ndarray, features: list[str], l2: float = 0.01) -> CoxModel:
    """Cox proportional hazards with Breslow ties and a small ridge penalty."""
    X = np.asarray(X, float)
    time, event = np.asarray(time, float), np.asarray(event, float)
    mean = np.nanmean(X, axis=0)
    scale = np.nanstd(X, axis=0)
    scale[scale == 0] = 1.0
    Z = np.where(np.isfinite(X), (X - mean) / scale, 0.0)
    order = np.argsort(-time, kind="stable")  # descending time -> cumulative sums give risk sets
    Z, t, e = Z[order], time[order], event[order]

    def nll(beta):
        eta = Z @ beta
        m = eta.max()
        w = np.exp(eta - m)
        cum_w = np.cumsum(w)
        cum_wz = np.cumsum(w[:, None] * Z, axis=0)
        # Breslow: all tied subjects share the risk set at the *last* index with that time
        _, first_idx = np.unique(-t, return_index=True)
        last_idx = np.r_[first_idx[1:] - 1, len(t) - 1]
        grp = np.searchsorted(first_idx, np.arange(len(t)), side="right") - 1
        rs_w = cum_w[last_idx][grp]
        rs_wz = cum_wz[last_idx][grp]
        ll = np.sum(e * (eta - m - np.log(rs_w)))
        grad = np.sum(e[:, None] * (Z - rs_wz / rs_w[:, None]), axis=0)
        return -ll + l2 * beta @ beta, -grad + 2 * l2 * beta

    res = minimize(nll, np.zeros(Z.shape[1]), jac=True, method="L-BFGS-B")
    beta = res.x
    # Breslow baseline cumulative hazard
    eta = (np.where(np.isfinite(X), (X - mean) / scale, 0.0)) @ beta
    w = np.exp(eta)
    uniq = np.unique(time[event == 1])
    cumhaz, h = [], 0.0
    for u in uniq:
        d = np.sum((time == u) & (event == 1))
        h += d / np.sum(w[time >= u])
        cumhaz.append(h)
    return CoxModel(list(features), beta, mean, scale, uniq, np.asarray(cumhaz))


def concordance_index(time, event, risk) -> float:
    """Harrell's C: share of comparable pairs where the higher risk failed first."""
    time, event, risk = (np.asarray(a, float) for a in (time, event, risk))
    ok = np.isfinite(time) & np.isfinite(event) & np.isfinite(risk)
    time, event, risk = time[ok], event[ok], risk[ok]
    conc, total = 0.0, 0.0
    for i in np.where(event == 1)[0]:
        later = time > time[i]
        n = later.sum()
        if n == 0:
            continue
        total += n
        conc += np.sum(risk[i] > risk[later]) + 0.5 * np.sum(risk[i] == risk[later])
    return float(conc / total) if total else float("nan")


def kaplan_meier(time, event) -> tuple[np.ndarray, np.ndarray]:
    time, event = np.asarray(time, float), np.asarray(event, float)
    uniq = np.unique(time[event == 1])
    surv, s = [], 1.0
    for u in uniq:
        at_risk = np.sum(time >= u)
        d = np.sum((time == u) & (event == 1))
        s *= 1 - d / at_risk
        surv.append(s)
    return np.r_[0.0, uniq], np.r_[1.0, surv]
