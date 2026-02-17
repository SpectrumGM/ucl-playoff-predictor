import pandas as pd
import numpy as np
from elo import find_elo

MATCHES = [
    ("Monaco", "Paris Saint-Germain"),
    ("Galatasaray", "Juventus"),
    ("Benfica", "Real Madrid"),
    ("Borussia Dortmund", "Atalanta"),
    ("Qarabağ", "Newcastle United"),
    ("Club Brugge", "Atlético Madrid"),
    ("Bodø/Glimt", "Inter"),
    ("Olympiacos", "Bayer Leverkusen"),
]

LEAGUE_POS = {
    "Real Madrid": 9, "Inter": 10, "Paris Saint-Germain": 11,
    "Newcastle United": 12, "Atalanta": 13, "Atlético Madrid": 14,
    "Juventus": 15, "Bayer Leverkusen": 16, "Club Brugge": 17,
    "Olympiacos": 18, "Borussia Dortmund": 19, "Galatasaray": 20,
    "Monaco": 21, "Qarabağ": 22, "Bodø/Glimt": 23, "Benfica": 24,
}

QUALITATIVE = {
    "Monaco": {"inj": 0.15, "home_atm": 0.55, "ko_exp": 0.30, "depth": 0.55, "form_q": 0.50, "mgr": 0.55},
    "Paris Saint-Germain": {"inj": 0.90, "home_atm": 0.75, "ko_exp": 0.85, "depth": 0.90, "form_q": 0.45, "mgr": 0.80},
    "Galatasaray": {"inj": 0.20, "home_atm": 0.95, "ko_exp": 0.40, "depth": 0.55, "form_q": 0.35, "mgr": 0.50},
    "Juventus": {"inj": 0.65, "home_atm": 0.80, "ko_exp": 0.70, "depth": 0.75, "form_q": 0.65, "mgr": 0.60},
    "Benfica": {"inj": 0.35, "home_atm": 0.80, "ko_exp": 0.50, "depth": 0.60, "form_q": 0.60, "mgr": 0.90},
    "Real Madrid": {"inj": 0.55, "home_atm": 0.90, "ko_exp": 0.90, "depth": 0.80, "form_q": 0.50, "mgr": 0.50},
    "Borussia Dortmund": {"inj": 0.30, "home_atm": 0.85, "ko_exp": 0.65, "depth": 0.65, "form_q": 0.40, "mgr": 0.55},
    "Atalanta": {"inj": 0.05, "home_atm": 0.75, "ko_exp": 0.55, "depth": 0.65, "form_q": 0.75, "mgr": 0.80},
    "Qarabağ": {"inj": 0.20, "home_atm": 0.60, "ko_exp": 0.15, "depth": 0.30, "form_q": 0.30, "mgr": 0.35},
    "Newcastle United": {"inj": 0.20, "home_atm": 0.85, "ko_exp": 0.45, "depth": 0.70, "form_q": 0.65, "mgr": 0.70},
    "Club Brugge": {"inj": 0.15, "home_atm": 0.65, "ko_exp": 0.35, "depth": 0.45, "form_q": 0.50, "mgr": 0.50},
    "Atlético Madrid": {"inj": 0.25, "home_atm": 0.80, "ko_exp": 0.80, "depth": 0.75, "form_q": 0.55, "mgr": 0.85},
    "Bodø/Glimt": {"inj": 0.10, "home_atm": 0.75, "ko_exp": 0.10, "depth": 0.35, "form_q": 0.45, "mgr": 0.45},
    "Inter": {"inj": 0.15, "home_atm": 0.80, "ko_exp": 0.80, "depth": 0.80, "form_q": 0.60, "mgr": 0.85},
    "Olympiacos": {"inj": 0.20, "home_atm": 0.75, "ko_exp": 0.35, "depth": 0.45, "form_q": 0.60, "mgr": 0.55},
    "Bayer Leverkusen": {"inj": 0.20, "home_atm": 0.70, "ko_exp": 0.55, "depth": 0.70, "form_q": 0.50, "mgr": 0.80},
}


def get_stats(df, team, before_date, n=30):
    mask = (
        (df["home"].str.contains(team, case=False, na=False)) |
        (df["away"].str.contains(team, case=False, na=False))
    ) & (df["date"] < before_date)

    matches = df[mask].sort_values("date", ascending=False).head(n)
    if len(matches) == 0:
        return {"win_rate": 0.33, "goal_diff": 0.0, "form": 5, "cl_wr": 0.33}

    wins, gf, ga, form = 0, 0, 0, 0
    for i, (_, m) in enumerate(matches.iterrows()):
        is_home = team.lower() in m["home"].lower()
        g_for = m["gh"] if is_home else m["ga"]
        g_against = m["ga"] if is_home else m["gh"]
        won = (is_home and m["result"] == "H") or (not is_home and m["result"] == "A")
        drew = m["result"] == "D"

        if won: wins += 1
        gf += g_for
        ga += g_against
        if i < 5:
            form += 3 if won else (1 if drew else 0)

    cl = matches[matches["competition"] == "Champions League"]
    cl_wins = sum(
        1 for _, m in cl.iterrows()
        if (team.lower() in m["home"].lower() and m["result"] == "H") or
           (team.lower() in m["away"].lower() and m["result"] == "A")
    )

    total = len(matches)
    return {
        "win_rate": wins / total,
        "goal_diff": (gf - ga) / total,
        "form": form,
        "cl_wr": cl_wins / len(cl) if len(cl) > 0 else 0.33,
    }


def get_h2h(df, team1, team2, before_date):
    cutoff = before_date - pd.DateOffset(years=10)
    mask = (
        ((df["home"].str.contains(team1, case=False, na=False)) &
         (df["away"].str.contains(team2, case=False, na=False))) |
        ((df["home"].str.contains(team2, case=False, na=False)) &
         (df["away"].str.contains(team1, case=False, na=False)))
    ) & (df["date"] < before_date) & (df["date"] > cutoff)

    h2h = df[mask]
    if len(h2h) == 0:
        return {"h2h_wr": 0.5, "h2h_gd": 0.0}

    wins, gf, ga = 0, 0, 0
    for _, m in h2h.iterrows():
        t1_home = team1.lower() in m["home"].lower()
        g_for = m["gh"] if t1_home else m["ga"]
        g_against = m["ga"] if t1_home else m["gh"]
        won = (t1_home and m["result"] == "H") or (not t1_home and m["result"] == "A")
        if won: wins += 1
        gf += g_for
        ga += g_against

    total = len(h2h)
    return {"h2h_wr": wins / total, "h2h_gd": (gf - ga) / total}


def build_features(df, elo):
    pred_date = pd.Timestamp("2026-02-17")
    all_features = []

    for unseeded, seeded in MATCHES:
        h_elo = find_elo(elo, unseeded)
        a_elo = find_elo(elo, seeded)
        h_stats = get_stats(df, unseeded, pred_date)
        a_stats = get_stats(df, seeded, pred_date)
        h2h = get_h2h(df, unseeded, seeded, pred_date)
        h_qual = QUALITATIVE.get(unseeded, {})
        a_qual = QUALITATIVE.get(seeded, {})
        h_pos = LEAGUE_POS.get(unseeded, 20)
        a_pos = LEAGUE_POS.get(seeded, 20)

        features = {
            "unseeded": unseeded,
            "seeded": seeded,
            "elo_diff": h_elo - a_elo,
            "wr_diff": h_stats["win_rate"] - a_stats["win_rate"],
            "gd_diff": h_stats["goal_diff"] - a_stats["goal_diff"],
            "form_diff": h_stats["form"] - a_stats["form"],
            "cl_wr_diff": h_stats["cl_wr"] - a_stats["cl_wr"],
            "h2h_wr": h2h["h2h_wr"],
            "h2h_gd": h2h["h2h_gd"],
            "inj_diff": a_qual.get("inj", 0.2) - h_qual.get("inj", 0.2),
            "home_atm_unseeded": h_qual.get("home_atm", 0.5),
            "home_atm_seeded": a_qual.get("home_atm", 0.5),
            "ko_exp_diff": h_qual.get("ko_exp", 0.3) - a_qual.get("ko_exp", 0.3),
            "depth_diff": h_qual.get("depth", 0.5) - a_qual.get("depth", 0.5),
            "form_q_diff": h_qual.get("form_q", 0.5) - a_qual.get("form_q", 0.5),
            "mgr_diff": h_qual.get("mgr", 0.5) - a_qual.get("mgr", 0.5),
            "lp_pos_diff": a_pos - h_pos,
        }
        all_features.append(features)

    return all_features
