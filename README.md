# UCL 2025-26 Knockout Playoff Predictor

Predicts which teams advance in the UEFA Champions League 2025-26 knockout playoff round using Elo ratings, historical stats, and Monte Carlo simulation.

## How it works

1. **Elo Ratings** — Calculated from 21,000+ matches across 5 leagues + Champions League (2014-2026). Same formula as chess: beat a strong team → gain more points.

2. **Feature Engineering** — 16 features per matchup:
   - Elo difference, win rates, goal difference, recent form
   - Champions League specific win rate
   - Head-to-head record (last 10 years)
   - Injuries, home atmosphere, knockout experience, squad depth, manager quality
   - League phase position difference

3. **Weighted Scoring** — Each feature is weighted based on football analytics research (FiveThirtyEight, Opta). Converted to probabilities via sigmoid function.

4. **Monte Carlo Simulation** — 100,000 simulations of both legs with realistic goal distributions. Accounts for:
   - Seeded team hosting the second leg (UEFA rule)
   - Extra time / penalties at 50-50 (slight home advantage)
   - No away goals rule (new format)

## Results

| Match | Prediction | Confidence |
|-------|-----------|------------|
| Monaco vs PSG | PSG advances | 87% |
| Qarabağ vs Newcastle | Newcastle advances | 86% |
| Bodø/Glimt vs Inter | Inter advances | 82% |
| Club Brugge vs Atlético | Atlético advances | 74% |
| Galatasaray vs Juventus | Juventus advances | 70% |
| Benfica vs Real Madrid | Real Madrid advances | 63% |
| Olympiacos vs Leverkusen | Leverkusen advances | 61% |
| Dortmund vs Atalanta | Atalanta advances | 55% |

## Setup

```
pip install pandas numpy scipy
```

Place CSV match data in `data/` folder:
- `england.csv`, `spain.csv`, `italy.csv`, `germany.csv`, `france.csv`
- `champions_league.csv`

Each CSV needs columns: `date`, `home`, `away`, `gh` (home goals), `ga` (away goals)

## Run

```
python predict.py
```

## Project Structure

```
├── analysis.ipynb   # Full walkthrough notebook (start here)
├── predict.py       # Main script
├── data.py          # Data loading
├── elo.py           # Elo rating calculation
├── features.py      # Feature engineering + qualitative data
├── model.py         # Weighted model + Monte Carlo simulation
├── requirements.txt
└── data/            # Match CSVs (not included)
```

## Notebook

`analysis.ipynb` contains the complete pipeline from data collection to predictions. It downloads match data directly from GitHub (openfootball), so no local CSVs needed — just run the cells.
