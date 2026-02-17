def calculate_elo(df):
    elo = {}
    df_sorted = df.sort_values("date").reset_index(drop=True)

    for _, match in df_sorted.iterrows():
        home, away = match["home"], match["away"]

        if home not in elo:
            elo[home] = 1500
        if away not in elo:
            elo[away] = 1500

        Ra, Rb = elo[home], elo[away]
        Ea = 1 / (1 + 10 ** ((Rb - Ra - 100) / 400))
        Eb = 1 - Ea

        if match["result"] == "H":
            Sa, Sb = 1, 0
        elif match["result"] == "A":
            Sa, Sb = 0, 1
        else:
            Sa, Sb = 0.5, 0.5

        K = 48 if match["competition"] == "Champions League" else 32

        gd = abs(match["gh"] - match["ga"])
        if gd == 2:
            K *= 1.5
        elif gd == 3:
            K *= 1.75
        elif gd >= 4:
            K *= 2.0

        elo[home] = Ra + K * (Sa - Ea)
        elo[away] = Rb + K * (Sb - Eb)

    return elo


def find_elo(elo_dict, team_name, default=1500):
    best = max(
        (v for k, v in elo_dict.items() if team_name.lower() in k.lower()),
        default=default
    )
    return best
