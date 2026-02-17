import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

LEAGUE_FILES = {
    "Premier League": "england.csv",
    "La Liga": "spain.csv",
    "Serie A": "italy.csv",
    "Bundesliga": "germany.csv",
    "Ligue 1": "france.csv",
    "Champions League": "champions_league.csv",
}


def load_data():
    frames = []
    for comp, filename in LEAGUE_FILES.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"hg": "gh", "ag": "ga"})
        if "gh" not in df.columns:
            for col in df.columns:
                if col in ("fthg", "home_goals", "hgoal"):
                    df = df.rename(columns={col: "gh"})
                if col in ("ftag", "away_goals", "agoal"):
                    df = df.rename(columns={col: "ga"})
        df["competition"] = comp
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"])
    df_all["gh"] = pd.to_numeric(df_all["gh"], errors="coerce")
    df_all["ga"] = pd.to_numeric(df_all["ga"], errors="coerce")
    df_all = df_all.dropna(subset=["date", "gh", "ga"])
    df_all["result"] = np.where(
        df_all["gh"] > df_all["ga"], "H",
        np.where(df_all["gh"] < df_all["ga"], "A", "D")
    )
    df_all = df_all.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df_all)} matches from {len(LEAGUE_FILES)} competitions")
    return df_all
