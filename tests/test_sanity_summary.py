import random

import pytest

from sanity_summary import bootstrap_ci, effective_share, summarize


def make_eval(correct, truncation=0.0, fmt=1.0):
    acc = sum(correct) / len(correct)
    return {
        "accuracy": acc, "format_rate": fmt, "accuracy_given_format": acc / fmt if fmt else 0.0,
        "mean_tokens": 100.0, "truncation_rate": truncation, "correct": correct,
    }


def make_report(seed, base_correct, final_correct, share_zero_std=0.4, truncation=0.05, model="m"):
    return {
        "seed": seed, "protocol": {"model": model},
        "baseline": make_eval(base_correct),
        "checkpoints": {"final": make_eval(final_correct, truncation=truncation)},
        "rl_log": [{"frac_reward_zero_std": share_zero_std, "kl": 0.01} for _ in range(10)],
    }


def improving(n=200, seed=0, p_base=0.4, p_final=0.6):
    rng = random.Random(seed)
    return [int(rng.random() < p_base) for _ in range(n)], [int(rng.random() < p_final) for _ in range(n)]


def test_pass_when_clearly_improving():
    reports = [make_report(s, *improving(seed=s)) for s in (1, 2, 3)]
    out = summarize(reports, n_boot=500)
    assert out["passed"], out["criteria"]
    assert out["gain_ci95_item_bootstrap"][0] > 0


def test_fail_when_no_improvement():
    reports = []
    for s in (1, 2, 3):
        base, _ = improving(seed=s)
        reports.append(make_report(s, base, list(base)))
    out = summarize(reports, n_boot=500)
    assert not out["passed"]
    assert not out["criteria"]["C1_ci_excludes_zero"]
    assert not out["criteria"]["C2_every_seed_improves"]


def test_fail_on_truncation_and_dead_groups():
    reports = [make_report(s, *improving(seed=s), truncation=0.5) for s in (1, 2)]
    assert not summarize(reports, n_boot=200)["criteria"]["C3_truncation_ok"]
    reports = [make_report(s, *improving(seed=s), share_zero_std=0.95) for s in (1, 2)]
    assert not summarize(reports, n_boot=200)["criteria"]["C4_enough_effective_updates"]


def test_rejects_mismatched_reports():
    a = make_report(1, *improving(n=100))
    b = make_report(2, *improving(n=50))
    with pytest.raises(ValueError):
        summarize([a, b])
    c = make_report(2, *improving(n=100), model="other")
    with pytest.raises(ValueError):
        summarize([a, c])
    with pytest.raises(ValueError):
        summarize([a, make_report(1, *improving(n=100))])


def test_helpers():
    assert effective_share([{"frac_reward_zero_std": 0.25}, {"frac_reward_zero_std": 0.75}]) == 0.5
    assert effective_share([]) is None
    low, high = bootstrap_ci([1.0] * 20, n_boot=100)
    assert low == high == 1.0
