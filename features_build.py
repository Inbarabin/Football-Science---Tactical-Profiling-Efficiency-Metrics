"""
Football Science – Feature Engineering

This module takes raw merged FBref + Understat data
and builds advanced tactical, offensive, and defensive features
for clustering and ML analysis.

Author: Inbar Rabin
"""

import soccerdata as sd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import dataframe_image as dfi
from pathlib import Path
from tqdm import tqdm


# Leagues supported in the project
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Seasons to process
SEASONS = [2020, 2021, 2022, 2023, 2024]


def feature_engineering(season=2024, league="ENG-Premier League"):
    """
    Generate advanced team-level features for a given league & season.

    Input:
    - Raw merged dataset (FBref + Understat)

    Output:
    - features_df: unscaled features
    - features_scaled_df: standardized features (for clustering)

    The features are grouped into:
    - Outcome / performance
    - Offensive efficiency
    - Defensive efficiency
    - Tactical style indicators
    """

    # -----------------------------
    # Load raw data
    # -----------------------------
    path = Path(f"data/raw/{league}/{league}_{season}.pkl")

    with open(path, "rb") as f:
        raw = pickle.load(f)

    # Safety check – fail fast if data is corrupted
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Loaded raw data is not a pandas DataFrame")

    # =========================
    # OUTCOME / PERFORMANCE
    # =========================
    raw["Matches"] = raw["W"] + raw["D"] + raw["L"]              # Total matches played
    raw["wins_perc(overall)"] = raw["W"] / raw["Matches"]       # Win rate across the season

    # Tactical attacking preference:
    # progressive carries vs progressive passes
    raw["att_style(tact)"] = raw["PrgC"] / raw["PrgP_"]

    # =========================
    # OFFENSIVE FEATURES
    # =========================

    # Finishing efficiency (actual goals vs expected goals)
    raw["attack_eff(off)"] = raw["goals"] / raw["xG"]

    # Conversion from shots on target
    raw["shot_conv(off)"] = raw["goals"] / raw["SoT"]

    # Shot accuracy
    raw["shot_eff(off)"] = raw["SoT"] / raw["shots"]

    # Chance creation per possession
    raw["poss_eff(off)"] = raw["xG"] / raw["Poss_"]

    # Productivity of progressive passing
    raw["progresive_eff(off)"] = raw["goals"] / raw["PrgP_"]

    # Carry-based attacking efficiency
    raw["carries_eff(off)"] = raw["goals"] / raw["CPA"]

    # Expected goal dominance
    raw["xG_balance(tact)"] = raw["xG"] - raw["xGA"]

    # Average possession share
    raw["possession(tact)"] = raw["Poss_"]

    # Performance relative to xG advantage
    # NOTE: sensitive to small xG differences
    raw["points_eff(tact)"] = raw["Pts"] / raw["xG_balance(tact)"]

    # Pressing effectiveness in attacking third
    raw["off_tackles_eff(off)"] = raw["Tackles_Att 3rd"] / raw["tackles_win"]

    # Combined style + efficiency metric
    raw["att_style_eff(off)"] = (
            raw["att_style(tact)"] * raw["poss_eff(off)"]
    )

    # =========================
    # DEFENSIVE FEATURES
    # =========================

    # Defensive over/under performance
    raw["def_eff(def)"] = raw["xGA"] / raw["goals_against"]

    # Pressing intensity (inverse PPDA)
    raw["ppda_eff(def)"] = 1 / raw["ppda"]

    # Deep completions conceded per match
    raw["deep_allowed_per_game(def)"] = (
            (1 / raw["deep_allowed"]) / raw["Matches"]
    )

    # Defensive resistance
    raw["tackles_goals_eff(def)"] = raw["tackles"] / raw["goals_against"]

    # Tackling success rate
    raw["tackles_eff(def)"] = raw["tackles_win"] / raw["tackles"]

    # Territorial dominance
    raw["deep_eff(tact)"] = raw["deep_completions"] / raw["deep_allowed"]

    # Offensive penetration adjusted by possession
    raw["deep_comp_pos_eff(tact)"] = (
            raw["deep_completions"] / raw["possession(tact)"]
    )

    # Duel success
    raw["challenge_eff(def)"] = raw["Challenges_Tkl%"]

    # Box defense efficiency
    raw["box_efficiency(def)"] = (
            raw["Clr_"] / (raw["deep_allowed"] + raw["Tackles_Def 3rd"])
    )

    # Defensive style indicator:
    # interceptions (proactive) vs blocks (reactive)
    raw["def_style(tact)"] = raw["Int_"] / raw["blocks"]

    # Error-related vulnerability
    raw["goal_leading_error(def)"] = 1 / raw["Err_"]

    # =========================
    # CLEANING
    # =========================

    # Replace invalid numeric results
    df = raw.replace([np.inf, -np.inf], np.nan)

    # Remove teams with undefined core attacking metrics
    df = df.dropna(subset=["attack_eff(off)"])

    # Conservative fill for remaining missing values
    df = df.fillna(0)

    # =========================
    # FEATURE SELECTION
    # =========================
    feature_cols = [
        "wins_perc(overall)",
        "attack_eff(off)",
        "shot_conv(off)",
        "shot_eff(off)",
        "poss_eff(off)",
        "progresive_eff(off)",
        "carries_eff(off)",
        "xG_balance(tact)",
        "possession(tact)",
        "points_eff(tact)",
        "off_tackles_eff(off)",
        "att_style_eff(off)",
        "def_eff(def)",
        "ppda_eff(def)",
        "deep_allowed_per_game(def)",
        "tackles_goals_eff(def)",
        "tackles_eff(def)",
        "deep_eff(tact)",
        "deep_comp_pos_eff(tact)",
        "challenge_eff(def)",
        "box_efficiency(def)",
        "def_style(tact)",
        "goal_leading_error(def)",
    ]

    # =========================
    # SCALING (for clustering)
    # =========================
    scaler = StandardScaler()
    features_scaled = df.copy()
    features_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    features_df = df[["team"] + feature_cols]
    features_scaled_df = features_scaled[["team"] + feature_cols]

    return features_df, features_scaled_df


def main():
    """
    Run feature engineering for all leagues & seasons.
    Saves both scaled and unscaled feature tables.
    """

    base_out_dir = Path("data/features")
    base_out_dir.mkdir(parents=True, exist_ok=True)

    for league in tqdm(LEAGUES, desc="Leagues"):
        league_dir = base_out_dir / league
        league_dir.mkdir(parents=True, exist_ok=True)

        for season in tqdm(SEASONS, desc=f"{league} seasons", leave=False):
            print(f"Processing {league} {season}")

            try:
                features_df, features_scaled_df = feature_engineering(
                    season=season,
                    league=league
                )

                with open(league_dir / f"{league}_{season}_features.pkl", "wb") as f:
                    pickle.dump(features_df, f)

                with open(league_dir / f"{league}_{season}_features_scaled.pkl", "wb") as f:
                    pickle.dump(features_scaled_df, f)

            except Exception as e:
                print(f"Failed for {league} {season}")
                print(e)


if __name__ == "__main__":
    main()
