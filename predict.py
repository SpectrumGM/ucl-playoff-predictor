import pandas as pd
import numpy as np
from scipy.special import expit

from data import load_data
from elo import calculate_elo
from features import get_stats, get_h2h, build_features
from model import predict_playoffs


def main():
    df = load_data()
    elo = calculate_elo(df)
    features = build_features(df, elo)
    results = predict_playoffs(features)

    print("\n🏆 UCL 2025-26 KNOCKOUT PLAYOFF PREDICTIONS")
    print("=" * 65)
    print("Format: Two legs, aggregate score, no away goals rule")
    print("Seeded team (9-16) hosts the second leg\n")

    results.sort(key=lambda x: max(x["t1_pct"], x["t2_pct"]), reverse=True)

    for i, r in enumerate(results, 1):
        fav = r["seeded"] if r["t2_pct"] > r["t1_pct"] else r["unseeded"]
        fav_pct = max(r["t1_pct"], r["t2_pct"])
        conf = "🔴" if fav_pct > 0.70 else "🟡" if fav_pct > 0.58 else "🟢"

        bar1 = "█" * int(r["t1_pct"] * 30)
        bar2 = "░" * int(r["t2_pct"] * 30)

        print(f" {i}. {r['unseeded']} vs {r['seeded']}")
        print(f"    {bar1}{bar2}")
        print(f"    {r['unseeded']}: {r['t1_pct']:.1%}  vs  {r['seeded']}: {r['t2_pct']:.1%}")
        print(f"    1st leg ({r['unseeded']} home): {r['leg1']}")
        print(f"    2nd leg ({r['seeded']} home):  {r['leg2']}")
        print(f"    {conf} {fav} advances ({fav_pct:.0%})\n")

    print("🔴 >70%  🟡 58-70%  🟢 <58%")
    print("100,000 Monte Carlo simulations")


if __name__ == "__main__":
    main()
