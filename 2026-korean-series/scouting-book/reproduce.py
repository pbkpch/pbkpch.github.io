#!/usr/bin/env python3
"""KS26 Samsung vs KT: team-specific scouting calculations, 2026-11-05 snapshot.

All inputs are public 2026 regular-season counts. See ../scouting-book/ for sources.
This is a replication / falsification aid, NOT a calibrated forecasting model.
Python 3.9+; no external dependencies.
"""
from __future__ import annotations
from math import comb
import json

GAMES_TOTAL = 144
H2H_GAMES = 16
SAM_HR_SEASON = 140
SAM_HR_VS_KT = 7
KT_HR_SEASON = 121
KT_HR_VS_SAM = 12
SAM_RISP_PA, SAM_RISP_OPS = 1656, .819
KT_RISP_PA, KT_RISP_OPS = 1721, .832
SAM_2OUT_RISP_PA, SAM_2OUT_RISP_OPS = 753, .728
KT_2OUT_RISP_PA, KT_2OUT_RISP_OPS = 798, .862

def pbinom(i: int, n: int, p: float) -> float:
    return comb(n, i) * p**i * (1-p)**(n-i)

def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k), with a recurrence stable in the p range used here."""
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    term = (1-p)**n
    total = term
    for i in range(k):
        term *= ((n-i)/(i+1)) * (p/(1-p))
        total += term
    return total

def exact_binomial_interval(k: int, n: int, alpha: float = .05):
    """Clopper-Pearson 95% for the conditional Poisson event proportion."""
    left, right = 0.0, 1.0
    for _ in range(110):
        mid = (left + right) / 2
        if binom_cdf(k-1, n, mid) > 1-alpha/2:
            left = mid
        else:
            right = mid
    lower = (left + right) / 2
    left, right = 0.0, 1.0
    for _ in range(110):
        mid = (left + right) / 2
        if binom_cdf(k, n, mid) > alpha/2:
            left = mid
        else:
            right = mid
    upper = (left + right) / 2
    return lower, upper

def rate_ratio(exposed_count: int, other_count: int,
               exposed_games: int = 16, other_games: int = 128):
    ratio = (exposed_count / exposed_games) / (other_count / other_games)
    p_lo, p_hi = exact_binomial_interval(exposed_count, exposed_count + other_count)
    exposure_factor = other_games / exposed_games
    rr_lo = exposure_factor * p_lo / (1-p_lo)
    rr_hi = exposure_factor * p_hi / (1-p_hi)
    return ratio, rr_lo, rr_hi

def conditional_binom_two_sided(k: int, n: int, p: float) -> float:
    """Two-sided probability-ordering test; exploratory, not multiplicity adjusted."""
    observed = pbinom(k, n, p)
    return sum(pbinom(i, n, p) for i in range(n+1)
               if pbinom(i, n, p) <= observed * (1 + 1e-12))

def lineup_swap_onbase(obp_current_2: float, obp_current_3: float,
                      pa_gap_per_game: float = .11, season_games: int = 144):
    """Additional times on base expected in a *hypothetical* constant-OBP swap.
    It is not expected runs, lineup optimization, or a causal team effect.
    """
    return (obp_current_3 - obp_current_2) * pa_gap_per_game * season_games

def report():
    sam_others = SAM_HR_SEASON - SAM_HR_VS_KT
    kt_others = KT_HR_SEASON - KT_HR_VS_SAM
    sam_ratio = rate_ratio(SAM_HR_VS_KT, sam_others)
    kt_ratio = rate_ratio(KT_HR_VS_SAM, kt_others)
    result = {
        "snapshot": "2026-11-05 before Korean Series Game 3",
        "observed": {
            "samsung_homeruns":{"vs_KT":SAM_HR_VS_KT,"vs_KT_games":16,
                               "vs_other_nine_teams":sam_others,"other_games":128},
            "kt_homeruns":{"vs_Samsung":KT_HR_VS_SAM,"vs_Samsung_games":16,
                           "vs_other_nine_teams":kt_others,"other_games":128},
            "overall_RISP":{"samsung":{"PA":SAM_RISP_PA,"OPS":SAM_RISP_OPS},
                            "kt":{"PA":KT_RISP_PA,"OPS":KT_RISP_OPS}},
            "two_out_RISP":{"samsung":{"PA":SAM_2OUT_RISP_PA,"OPS":SAM_2OUT_RISP_OPS},
                            "kt":{"PA":KT_2OUT_RISP_PA,"OPS":KT_2OUT_RISP_OPS}},
        },
        "calculated": {
            "samsung_vs_KT_HR_per_game": SAM_HR_VS_KT / H2H_GAMES,
            "samsung_vs_other_HR_per_game": sam_others / (GAMES_TOTAL-H2H_GAMES),
            "samsung_HR_rate_ratio_and_exact_95": sam_ratio,
            "samsung_exploratory_two_sided_p": conditional_binom_two_sided(
                SAM_HR_VS_KT, SAM_HR_SEASON, H2H_GAMES/GAMES_TOTAL),
            "kt_HR_rate_ratio_and_exact_95": kt_ratio,
            "two_out_RISP_OPS_KT_minus_Samsung": KT_2OUT_RISP_OPS-SAM_2OUT_RISP_OPS,
            "lineup_swap_Samsung_extra_onbases_144G": lineup_swap_onbase(.375,.458),
            "lineup_swap_KT_extra_onbases_144G": lineup_swap_onbase(.342,.441),
            "hilliard_share_of_KT_HRs":44/KT_HR_SEASON,
            "diaz_share_of_Samsung_HRs":26/SAM_HR_SEASON,
        },
        "methods":{
            "HR_exact_interval":"Conditional Poisson -> binomial event proportion exact Clopper-Pearson interval, transformed to rate ratio",
            "lineup":"Constant OBP and hypothetical .11 PA/slot/game; not expected runs",
            "warning":"Multiple comparisons, park/lineup/PA not adjusted; not causal; no p-value correction; situational OPS provider uses PA-approx OBP."
        }
    }
    assert abs(sam_ratio[0] - .42105263157894735) < 1e-10
    assert abs(sam_ratio[1] - .16606618) < 1e-5
    assert abs(sam_ratio[2] - .89190161) < 1e-5
    assert abs(result["calculated"]["two_out_RISP_OPS_KT_minus_Samsung"]-.134) < 1e-12
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    report()
