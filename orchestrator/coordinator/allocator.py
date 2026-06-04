"""Epoch allocator: bandit / optimal-stopping range allocation.

Runs ONCE PER EPOCH off the GPU hot path (README "Telemetry -> bandit";
ADR-0004; ``docs/research/bandit-allocation.md``). It decides where to point the
next block of GPU-hours and when to *stop* a losing arm.

Two arm families, per the research note:

Brute-force arms (no exposed pubkey)
    Hit is a memoryless Poisson process in swept keys: marginal hit-probability
    per key is constant ``1/R`` while unswept, so there is **no learning signal**
    and therefore **no exploration term**. Ranking is pure exploitation:

        index_i = prize_i / R_i * shrinkage(contention_i)

    where ``R_i`` is the (remaining) range size and ``shrinkage`` discounts a
    contended range. We also compute a hazard-adjusted expected value and apply
    a **stop-loss**: an arm whose ``EV_i < 0`` is dropped from the ranking. This
    is the part the spec requires to be *fully* implemented and tested:
    a narrower range (smaller ``R``) must rank strictly higher than a wider one
    when contention is equal.

Kangaroo arms (exposed pubkey)
    Expected group-ops to collision ``~ c*sqrt(2^b)``. DP statistics give a
    real-time progress signal, so this arm *does* support learning -> a
    Bayesian / Gittins-index treatment is meaningful. Implemented here as a
    documented SKELETON: a Gamma-posterior on the collision *rate* updated from
    observed DP rate, from which we derive a Gittins-style index. Marked clearly
    as a skeleton; the full kangaroo solver is a kernel-layer concern.

Honest scope (ADR-0004): this is an *allocation* optimizer + stop-loss. It does
NOT beat sqrt(n) and does NOT make brute force tractable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Shrinkage on contention
# ---------------------------------------------------------------------------


def shrinkage(contention: float) -> float:
    """Multiplicative discount in (0, 1] for a contended range.

    ``contention`` is a unit-free intensity in [0, inf): 0 = uncontested
    (factor 1.0), larger = more pools grinding it. We use ``exp(-contention)``,
    a smooth, monotone-decreasing shrinkage that never inverts the ordering of
    two arms with *equal* contention. This is the ``S(contention_i)`` term in
    the research note's policy ``prize_i / R_i * S(contention_i)``.
    """
    if contention < 0:
        raise ValueError("contention must be >= 0")
    return math.exp(-contention)


# ---------------------------------------------------------------------------
# Brute-force arm
# ---------------------------------------------------------------------------


@dataclass
class BruteForceArm:
    """A candidate brute-force range (no exposed pubkey).

    Parameters
    ----------
    name:
        Arm identifier.
    range_size:
        ``R_i`` — number of *remaining* (unswept) candidate keys. Smaller is
        better: higher per-key marginal value ``1/R``.
    prize:
        ``prize_i`` — payoff if solved (e.g. puzzle reward), in arbitrary units.
    contention:
        Contention intensity (>= 0) feeding :func:`shrinkage`.
    hazard:
        Exponential front-run hazard rate ``h_i`` (per epoch). Folds into the
        hazard-adjusted EV used by the stop-loss.
    sweep_rate:
        Measured keys/s for the budgeted hardware (telemetry-fed). Used only by
        :meth:`expected_value` to convert a time budget into swept keys.
    """

    name: str
    range_size: int
    prize: float = 1.0
    contention: float = 0.0
    hazard: float = 0.0
    sweep_rate: float = 1.0

    def index(self) -> float:
        """Exploitation index ``prize / R * shrinkage(contention)``.

        No exploration term: there is no mid-sweep learning signal for a
        memoryless uniform search (ADR-0004 / research note).
        """
        if self.range_size <= 0:
            raise ValueError("range_size must be positive")
        return (self.prize / self.range_size) * shrinkage(self.contention)

    def expected_value(self, budget_seconds: float, cost_per_second: float) -> float:
        """Hazard-adjusted EV of spending ``budget_seconds`` on this arm.

        ``EV = P(we solve in dt) * prize * P(not front-run | we solve) - cost``.

        * ``P(we solve in dt)``: fraction of the remaining range we can sweep in
          the budget, capped at 1 — ``min(1, sweep_rate*dt / R)`` (uniform key
          location over the unswept portion).
        * ``P(not front-run | we solve) = exp(-hazard * dt)``: the exponential
          contention hazard from the research note.
        * ``cost = cost_per_second * dt``.
        """
        if self.range_size <= 0:
            raise ValueError("range_size must be positive")
        keys_sweepable = self.sweep_rate * budget_seconds
        p_solve = min(1.0, keys_sweepable / self.range_size)
        p_not_frontrun = math.exp(-self.hazard * budget_seconds)
        reward = p_solve * self.prize * p_not_frontrun
        cost = cost_per_second * budget_seconds
        return reward - cost


def rank_brute_force(
    arms: List[BruteForceArm],
    budget_seconds: float = 1.0,
    cost_per_second: float = 0.0,
    apply_stop_loss: bool = True,
) -> List[Tuple[BruteForceArm, float]]:
    """Rank brute-force arms by exploitation index, high to low.

    When ``apply_stop_loss`` is set, any arm whose hazard-adjusted
    :meth:`BruteForceArm.expected_value` is negative is dropped (the optimal-
    stopping rule: quit a losing arm). With the default zero cost the stop-loss
    is a no-op and pure ``prize/R*shrinkage`` ordering is returned.

    Returns a list of ``(arm, index)`` sorted by index descending. Ties break on
    arm name for determinism.

    Guarantee exercised by the tests: with equal contention, a narrower range
    (smaller ``R``) ranks **strictly higher** than a wider one.
    """
    kept: List[Tuple[BruteForceArm, float]] = []
    for arm in arms:
        if apply_stop_loss and cost_per_second > 0:
            if arm.expected_value(budget_seconds, cost_per_second) < 0:
                continue  # stop-loss: hazard-adjusted EV < 0, drop the arm
        kept.append((arm, arm.index()))
    kept.sort(key=lambda pair: (-pair[1], pair[0].name))
    return kept


# ---------------------------------------------------------------------------
# Kangaroo arm — Bayesian / Gittins-index SKELETON
# ---------------------------------------------------------------------------


@dataclass
class KangarooArm:
    """A candidate kangaroo range (exposed pubkey). SKELETON.

    Expected group-ops to collision ``~ c*sqrt(2^b)`` (c ~ 1.7 for SOTA
    3-kangaroo). DP statistics provide a real-time progress signal, so a
    Bayesian/Gittins treatment is meaningful here (unlike brute force).

    We keep a conjugate **Gamma posterior on the collision rate** ``mu`` (events
    per epoch). Prior ``Gamma(alpha0, beta0)``. Each epoch we observe an
    effective number of DP "events" over an exposure; the posterior updates as
    ``alpha += events``, ``beta += exposure``. The posterior-mean rate is
    ``alpha/beta`` and the implied expected time-to-collision is
    ``c*sqrt(2^b) / (rate * dp_yield)``.

    SKELETON SCOPE: the mapping from raw DP rate to collision-rate "events" and
    the herd-overlap correction are simplified placeholders. The full treatment
    (and the actual walk) lives in the kernel layer. The public API and the
    posterior update are real so the coordinator can wire telemetry in.
    """

    name: str
    interval_bits: int  # b in 2^b
    prize: float = 1.0
    c: float = 1.7  # SOTA 3-kangaroo constant
    contention: float = 0.0
    hazard: float = 0.0
    # Gamma posterior on collision rate (events/epoch).
    alpha: float = 1.0  # prior shape
    beta: float = 1.0  # prior rate
    dp_yield: float = 1.0  # DPs observed per expected collision-progress unit

    def expected_group_ops(self) -> float:
        """``c * sqrt(2^b)`` — expected group operations to collision."""
        return self.c * math.sqrt(2.0**self.interval_bits)

    def update_from_dp_rate(self, observed_dp_rate: float, exposure: float = 1.0):
        """Conjugate posterior update from an epoch's observed DP rate.

        Treat ``observed_dp_rate * exposure`` as an event count and ``exposure``
        as the time the herd ran. SKELETON: real code would weight by herd
        overlap and DP threshold ``d``.
        """
        if observed_dp_rate < 0 or exposure <= 0:
            raise ValueError("bad telemetry")
        self.alpha += observed_dp_rate * exposure
        self.beta += exposure

    def posterior_rate(self) -> float:
        """Posterior-mean collision-progress rate ``alpha / beta``."""
        return self.alpha / self.beta

    def expected_time_to_collision(self) -> float:
        """Expected epochs to collision from the current posterior.

        ``E[group_ops] / (rate * dp_yield)``; +inf if the rate has collapsed to
        zero (no progress signal yet).
        """
        rate = self.posterior_rate() * self.dp_yield
        if rate <= 0:
            return math.inf
        return self.expected_group_ops() / rate

    def gittins_index(self) -> float:
        """Gittins-style allocation index (SKELETON).

        Higher is better. We use ``prize * shrinkage(contention) /
        E[time_to_collision]`` — value per unit expected remaining time — which
        is the Gittins-index spirit (reward rate of the optimal continuation)
        for an arm whose posterior favours faster collision. A rigorous Gittins
        index requires solving the dynamic-allocation index for the Gamma
        bandit; that calibration is intentionally deferred (skeleton).
        """
        ettc = self.expected_time_to_collision()
        if ettc == math.inf:
            return 0.0
        return self.prize * shrinkage(self.contention) / ettc


# ---------------------------------------------------------------------------
# Cross-arm allocator
# ---------------------------------------------------------------------------


@dataclass
class Allocator:
    """Holds the candidate arms and computes the per-epoch allocation.

    The coordinator constructs one of these, registers arms, feeds telemetry
    (sweep rates, DP rates) each epoch, and calls :meth:`allocate` to get the
    argmax arm to steer idle capacity toward (the README's ``ReallocHint``).
    """

    brute_force: List[BruteForceArm] = field(default_factory=list)
    kangaroo: List[KangarooArm] = field(default_factory=list)
    halted: bool = False  # set True on FOUND -> allocation stops

    def add_brute_force(self, arm: BruteForceArm) -> None:
        self.brute_force.append(arm)

    def add_kangaroo(self, arm: KangarooArm) -> None:
        self.kangaroo.append(arm)

    def halt(self) -> None:
        """Stop allocating (called on FOUND)."""
        self.halted = True

    def allocate(
        self, budget_seconds: float = 1.0, cost_per_second: float = 0.0
    ) -> Optional[str]:
        """Return the name of the argmax arm to fund this epoch, or None.

        Returns ``None`` if halted (FOUND) or if no arm survives the stop-loss.
        Brute-force exploitation indices and kangaroo Gittins indices are placed
        on a comparable footing via the same ``prize/.. * shrinkage`` shape, and
        the global argmax wins (README "allocate next GPU-block to argmax arm").
        """
        if self.halted:
            return None
        ranked_bf = rank_brute_force(
            self.brute_force,
            budget_seconds=budget_seconds,
            cost_per_second=cost_per_second,
        )
        candidates: List[Tuple[str, float]] = [
            (arm.name, idx) for arm, idx in ranked_bf
        ]
        for arm in self.kangaroo:
            candidates.append((arm.name, arm.gittins_index()))
        if not candidates:
            return None
        candidates.sort(key=lambda pair: (-pair[1], pair[0]))
        best_name, best_idx = candidates[0]
        if best_idx <= 0:
            return None
        return best_name
