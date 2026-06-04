"""Allocator tests: narrower range ranks higher (d), stop-loss, kangaroo skeleton."""

from __future__ import annotations

import math

import pytest

from orchestrator.coordinator.allocator import (
    Allocator,
    BruteForceArm,
    KangarooArm,
    rank_brute_force,
    shrinkage,
)


# -- (d) narrower range ranks STRICTLY higher, contention equal -------------


def test_narrower_range_ranks_strictly_higher_equal_contention():
    wide = BruteForceArm(name="wide", range_size=2**40, prize=1.0, contention=0.5)
    narrow = BruteForceArm(name="narrow", range_size=2**20, prize=1.0, contention=0.5)
    ranked = rank_brute_force([wide, narrow])
    names = [arm.name for arm, _ in ranked]
    assert names[0] == "narrow", names
    # Strictly higher index, not merely >=.
    idx = {arm.name: score for arm, score in ranked}
    assert idx["narrow"] > idx["wide"]


def test_ranking_orders_multiple_arms_by_size_when_equal_otherwise():
    arms = [
        BruteForceArm(name="r2_30", range_size=2**30, prize=1.0, contention=0.2),
        BruteForceArm(name="r2_20", range_size=2**20, prize=1.0, contention=0.2),
        BruteForceArm(name="r2_25", range_size=2**25, prize=1.0, contention=0.2),
    ]
    ranked = rank_brute_force(arms)
    assert [a.name for a, _ in ranked] == ["r2_20", "r2_25", "r2_30"]


def test_prize_breaks_against_size():
    # A 4x bigger range with >4x prize should win on prize/R.
    small = BruteForceArm(name="small", range_size=2**20, prize=1.0)
    big_rich = BruteForceArm(name="big_rich", range_size=2**22, prize=8.0)
    ranked = rank_brute_force([small, big_rich])
    assert ranked[0][0].name == "big_rich"


def test_shrinkage_monotone_and_bounded():
    assert shrinkage(0.0) == pytest.approx(1.0)
    assert 0.0 < shrinkage(5.0) < shrinkage(1.0) < shrinkage(0.0)
    with pytest.raises(ValueError):
        shrinkage(-0.1)


def test_higher_contention_lowers_rank_equal_size():
    quiet = BruteForceArm(name="quiet", range_size=2**24, prize=1.0, contention=0.0)
    busy = BruteForceArm(name="busy", range_size=2**24, prize=1.0, contention=2.0)
    ranked = rank_brute_force([busy, quiet])
    assert ranked[0][0].name == "quiet"


# -- stop-loss (hazard-adjusted EV < 0 drops the arm) -----------------------


def test_stop_loss_drops_negative_ev_arm():
    # High hazard + tiny solve probability + real cost => EV < 0 => dropped.
    losing = BruteForceArm(
        name="losing",
        range_size=10**18,  # essentially unsearchable in the budget
        prize=1.0,
        hazard=5.0,
        sweep_rate=1.0,
    )
    winning = BruteForceArm(
        name="winning",
        range_size=10,  # tiny, swept in the budget
        prize=1.0,
        hazard=0.0,
        sweep_rate=100.0,
    )
    ranked = rank_brute_force(
        [losing, winning], budget_seconds=1.0, cost_per_second=0.5
    )
    names = [a.name for a, _ in ranked]
    assert "losing" not in names
    assert "winning" in names


def test_stop_loss_noop_when_cost_zero():
    arm = BruteForceArm(name="a", range_size=10**18, prize=1.0, hazard=5.0)
    ranked = rank_brute_force([arm], budget_seconds=1.0, cost_per_second=0.0)
    assert [a.name for a, _ in ranked] == ["a"]


def test_expected_value_signs():
    arm = BruteForceArm(
        name="ev", range_size=100, prize=10.0, hazard=0.0, sweep_rate=200.0
    )
    # sweep_rate*dt=200 >= R=100 -> p_solve=1, no hazard -> EV = prize - cost.
    assert arm.expected_value(1.0, cost_per_second=1.0) == pytest.approx(9.0)
    # With cost above prize, EV goes negative.
    assert arm.expected_value(1.0, cost_per_second=20.0) < 0


# -- kangaroo Gittins / Bayesian skeleton -----------------------------------


def test_kangaroo_posterior_updates_increase_rate():
    arm = KangarooArm(name="k", interval_bits=40)
    base_rate = arm.posterior_rate()
    arm.update_from_dp_rate(observed_dp_rate=100.0, exposure=1.0)
    assert arm.posterior_rate() > base_rate


def test_kangaroo_more_dps_shorter_time_to_collision():
    slow = KangarooArm(name="slow", interval_bits=40)
    fast = KangarooArm(name="fast", interval_bits=40)
    slow.update_from_dp_rate(1.0, exposure=1.0)
    fast.update_from_dp_rate(1000.0, exposure=1.0)
    assert fast.expected_time_to_collision() < slow.expected_time_to_collision()
    assert fast.gittins_index() > slow.gittins_index()


def test_kangaroo_expected_group_ops_scales_with_bits():
    small = KangarooArm(name="s", interval_bits=20)
    big = KangarooArm(name="b", interval_bits=40)
    # c*sqrt(2^b): 2^10 ratio between 2^20 and 2^40 group counts.
    assert big.expected_group_ops() / small.expected_group_ops() == pytest.approx(
        2.0**10
    )


def test_kangaroo_no_signal_yields_zero_index():
    arm = KangarooArm(name="k", interval_bits=40, alpha=0.0, beta=1.0)
    # rate=0 -> time-to-collision inf -> index 0 (don't fund a dead arm).
    assert arm.posterior_rate() == 0.0
    assert math.isinf(arm.expected_time_to_collision())
    assert arm.gittins_index() == 0.0


# -- cross-arm allocate + halt ----------------------------------------------


def test_allocate_picks_narrowest_brute_force_arm():
    alloc = Allocator()
    alloc.add_brute_force(BruteForceArm(name="wide", range_size=2**40))
    alloc.add_brute_force(BruteForceArm(name="narrow", range_size=2**20))
    assert alloc.allocate() == "narrow"


def test_allocate_halts_after_found():
    alloc = Allocator()
    alloc.add_brute_force(BruteForceArm(name="a", range_size=2**20))
    assert alloc.allocate() == "a"
    alloc.halt()
    assert alloc.allocate() is None
