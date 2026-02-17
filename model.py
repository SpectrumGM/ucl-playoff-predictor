import numpy as np
from scipy.special import expit

WEIGHTS = {
    "elo_diff": 0.0025,
    "wr_diff": 0.8,
    "gd_diff": 0.3,
    "form_diff": 0.04,
    "cl_wr_diff": 0.5,
    "h2h_wr": 0.4,
    "h2h_gd": 0.15,
    "inj_diff": 0.6,
    "ko_exp_diff": 0.3,
    "depth_diff": 0.2,
    "form_q_diff": 0.4,
    "mgr_diff": 0.3,
    "lp_pos_diff": 0.015,
}

HOME_ATM_WEIGHT = 0.5
SECOND_LEG_BONUS = 0.08
N_SIMULATIONS = 100_000

WIN_GOALS = np.array([1, 2, 3, 4])
WIN_PROBS = np.array([0.42, 0.33, 0.18, 0.07])
LOSE_GOALS = np.array([0, 1, 2])
LOSE_PROBS = np.array([0.55, 0.32, 0.13])
DRAW_GOALS = np.array([0, 1, 2, 3])
DRAW_PROBS = np.array([0.25, 0.42, 0.25, 0.08])


def neutral_score(f):
    score = 0
    for feat, w in WEIGHTS.items():
        if feat == "h2h_wr":
            score += (f[feat] - 0.5) * w
        else:
            score += f[feat] * w
    return score


def leg_probabilities(ns, home_atm, second_leg=False):
    total = ns + home_atm * HOME_ATM_WEIGHT
    if second_leg:
        total += SECOND_LEG_BONUS

    home_base = expit(total)
    draw = max(0.12, min(0.28, 0.24 - 0.05 * abs(total)))
    rem = 1 - draw
    return rem * home_base, draw, rem * (1 - home_base)


def simulate_leg(p_home, p_draw, rng):
    r = rng.random()
    if r < p_home:
        return (
            rng.choice(WIN_GOALS, p=WIN_PROBS),
            rng.choice(LOSE_GOALS, p=LOSE_PROBS),
        )
    elif r < p_home + p_draw:
        g = rng.choice(DRAW_GOALS, p=DRAW_PROBS)
        return g, g
    else:
        return (
            rng.choice(LOSE_GOALS, p=LOSE_PROBS),
            rng.choice(WIN_GOALS, p=WIN_PROBS),
        )


def predict_playoffs(features_list):
    rng = np.random.default_rng(42)
    results = []

    for f in features_list:
        ns = neutral_score(f)

        p1_h, p1_d, p1_a = leg_probabilities(ns, f["home_atm_unseeded"])
        p2_h, p2_d, p2_a = leg_probabilities(-ns, f["home_atm_seeded"], second_leg=True)

        unseeded_advances = 0

        for _ in range(N_SIMULATIONS):
            g1_uns, g1_seed = simulate_leg(p1_h, p1_d, rng)
            g2_seed, g2_uns = simulate_leg(p2_h, p2_d, rng)

            total_uns = g1_uns + g2_uns
            total_seed = g1_seed + g2_seed

            if total_uns > total_seed:
                unseeded_advances += 1
            elif total_uns == total_seed:
                unseeded_advances += rng.random() < 0.45

        t1 = unseeded_advances / N_SIMULATIONS
        t2 = 1 - t1

        results.append({
            "unseeded": f["unseeded"],
            "seeded": f["seeded"],
            "t1_pct": t1,
            "t2_pct": t2,
            "leg1": f"{f['unseeded'][:3]} {p1_h:.0%} | Draw {p1_d:.0%} | {f['seeded'][:3]} {p1_a:.0%}",
            "leg2": f"{f['seeded'][:3]} {p2_h:.0%} | Draw {p2_d:.0%} | {f['unseeded'][:3]} {p2_a:.0%}",
        })

    return results
