#!/usr/bin/env python3
"""2026 Korean Series Samsung–KT: reproducible pre-G2 arithmetic.

Scope: 2026 regular season (144 games each), KS G1 completed.
Data cutoff: 2026-11-04 before Game 2.
This is a scenario calculator, NOT a calibrated baseball forecast.

Primary sources:
KBO/KT club team records, Yagoonara team sabermetrics and KS G1/G2 box scores.
See the accompanying published article for links.
"""
from __future__ import annotations

from functools import lru_cache
from math import sqrt

SAM = dict(w=83, l=58, t=3, rs=806, ra=669)
KT = dict(w=90, l=49, t=5, rs=825, ra=649)
VENUE = {2: "KT", 3: "SAM", 4: "SAM", 5: "SAM", 6: "KT", 7: "KT"}
HOME_SHIFT = 0.015  # ±1.5 percentage points; explicit what-if assumption
PYTH_EXP = 1.83  # fixed exponent, not KBO-specific fitted value

def pyth(rs: int, ra: int) -> float:
    return rs**PYTH_EXP / (rs**PYTH_EXP + ra**PYTH_EXP)

def log5(p_s: float, p_k: float) -> float:
    return p_s * (1 - p_k) / (p_s * (1 - p_k) + (1 - p_s) * p_k)

def neutral_runs() -> float:
    return log5(pyth(SAM["rs"], SAM["ra"]), pyth(KT["rs"], KT["ra"]))

def neutral_record() -> float:
    return log5(SAM["w"] / (SAM["w"] + SAM["l"]),
                KT["w"] / (KT["w"] + KT["l"]))

def series_prob(p: float, home: float, game: int = 2,
                samsung_wins: int = 0, kt_wins: int = 1) -> float:
    """Exact probability of Samsung reaching four wins first."""
    @lru_cache(None)
    def recurse(g: int, s: int, k: int) -> float:
        if s >= 4:
            return 1.0
        if k >= 4 or g > 7:
            return 0.0
        q = p + (home if VENUE[g] == "SAM" else -home)
        if not 0 <= q <= 1:
            raise ValueError("Game-level win probability outside [0,1]")
        return q * recurse(g + 1, s + 1, k) + (1 - q) * recurse(g + 1, s, k + 1)
    return recurse(game, samsung_wins, kt_wins)

def wilson_interval(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval: simplified iid Bernoulli model, not skill CI."""
    p = x / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return center - half, center + half

def k_minus_bb(bf: int, strikeouts: int, walks: int) -> tuple[float, float]:
    """Rate and multinomial iid sampling variance for (K-BB)/BF."""
    p_k, p_bb = strikeouts / bf, walks / bf
    rate = p_k - p_bb
    var = (p_k + p_bb - rate**2) / bf
    return rate, var

def pitcher_diff_ci() -> tuple[float, float, float]:
    # Feddack minus Go Young-pyo, normal approximation with independent samples
    fed, v_fed = k_minus_bb(303, 80, 11)
    go, v_go = k_minus_bb(706, 183, 22)
    diff = fed - go
    half = 1.96 * sqrt(v_fed + v_go)
    return diff, diff - half, diff + half

def report(p: float) -> dict[str, float]:
    current = series_prob(p, HOME_SHIFT)
    tied = series_prob(p, HOME_SHIFT, 3, 1, 1)
    down02 = series_prob(p, HOME_SHIFT, 3, 0, 2)
    game2 = p - HOME_SHIFT
    assert abs(current - (game2 * tied + (1 - game2) * down02)) < 1e-12
    home_p = p + HOME_SHIFT
    return {
        "neutral_single_game": p,
        "g2_samsung_win": game2,
        "current_down_0_1": current,
        "after_g2_win_tied_1_1": tied,
        "after_g2_loss_down_0_2": down02,
        "g2_swing_tied_vs_down_0_2": tied - down02,
        "after_g2_win_daegu_sweep_to_title": home_p**3,
        "after_g2_win_daegu_two_or_more_wins": home_p**2 * (3 - 2*home_p),
    }

def main() -> None:
    import json
    result = {
        "data_cutoff": "2026-11-04 before Korean Series Game 2",
        "record_log5": report(neutral_record()),
        "runs_log5": report(neutral_runs()),
        "h2h_samsung_9_of_16_wilson_95": wilson_interval(9, 16),
        "k_minus_bb_feddack_minus_go_95_normal": pitcher_diff_ci(),
        "disclaimer": "Scenario arithmetic only. Not fitted/calibrated. Independence, "
                      "fixed home shift, unchanged roster/pitcher strength assumed.",
    }
    assert 0.235 < result["record_log5"]["current_down_0_1"] < 0.237
    assert 0.298 < result["runs_log5"]["current_down_0_1"] < 0.300
    assert 0.459 < result["runs_log5"]["after_g2_win_tied_1_1"] < 0.461
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
