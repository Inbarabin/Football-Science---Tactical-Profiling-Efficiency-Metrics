"""
Football Science – Feature Engineering Pipeline

This script builds season-level team features by combining:
- FBref team season statistics
- Understat team match statistics

The output is a clean, merged dataset per league & season,
ready for clustering / ML analysis.

Author: Inbar Rabin
"""

import soccerdata as sd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import dataframe_image as dfi
from name_map import apply_name_map
from pathlib import Path
from tqdm import tqdm


# Supported leagues (FBref + Understat naming)
LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# Seasons to process
SEASONS = [2020, 2021, 2022, 2023, 2024]


def create_features(season=2024, league="ENG-Premier League"):
    """
    Build a season-level efficiency table for all teams in a given league & season.

    Pipeline:
    1. Load raw data from FBref and Understat
    2. Clean and normalize column structure
    3. Aggregate match-level data to season-level
    4. Resolve team-name mismatches
    5. Merge sources and save raw feature table

    Parameters
    ----------
    season : int
        Season year (e.g. 2024)
    league : str
        League identifier compatible with soccerdata
    """

    # -----------------------------
    # STEP 1: Initialize data sources
    # -----------------------------
    fb = sd.FBref(leagues=league, seasons=season)
    us = sd.Understat(leagues=league, seasons=season)

    # NOTE: ClubElo imported for future extensions (rating-based features)
    elo = sd.ClubElo()

    # -----------------------------
    # STEP 2: Load FBref team season stats
    # -----------------------------
    df_fb = fb.read_team_season_stats(stat_type="standard")
    df_shooting = fb.read_team_season_stats(stat_type="shooting")
    df_passing = fb.read_team_season_stats(stat_type="passing")
    df_possession = fb.read_team_season_stats(stat_type="possession")
    df_defense = fb.read_team_season_stats(stat_type="defense")

    # Debug aid: inspect FBref team names
    print("\nFBref team names (from index):")
    print(sorted(df_fb.index.tolist()))

    # -----------------------------
    # Flatten MultiIndex columns
    # IMPORTANT: FBref tables often return MultiIndex columns
    # -----------------------------
    for df in [df_fb, df_shooting, df_passing, df_possession, df_defense]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # -----------------------------
    # Rename commonly used columns
    # (standardized naming across data sources)
    # -----------------------------
    df_fb.rename(columns={"Standard_Sh": "shots"}, inplace=True)

    df_shooting.rename(columns={
        "Standard_SoT": "SoT",
        "Standard_Sh": "shots",
    }, inplace=True)

    df_passing.rename(columns={"Total_Cmp": "Cmp"}, inplace=True)

    df_possession.rename(columns={
        "Carries_PrgC": "PrgC",
        "Carries_CPA": "CPA"
    }, inplace=True)

    df_defense.rename(columns={
        "Tackles_Tkl": "tackles",
        "Tackles_TklW": "tackles_win",
        "Blocks_Blocks": "blocks"
    }, inplace=True)

    # -----------------------------
    # Merge FBref stat tables
    # -----------------------------
    df_fb = (
        df_fb
        .merge(df_shooting, on="team", how="left", suffixes=("", "_shooting"))
        .merge(df_passing, on="team", how="left", suffixes=("", "_pass"))
        .merge(df_possession, on="team", how="left", suffixes=("", "_poss"))
        .merge(df_defense, on="team", how="left", suffixes=("", "_defense"))
    )

    # -----------------------------
    # STEP 3: Load Understat match-level data
    # -----------------------------
    df_us = us.read_team_match_stats()

    # Debug aid: inspect Understat naming
    print(sorted(df_us["home_team"].unique()))

    # -----------------------------
    # Build home & away tables
    # Each row represents one team in one match
    # -----------------------------
    home = df_us[[
        "home_team", "home_points", "home_goals", "away_goals", "home_xg", "away_xg",
        "home_ppda", "away_ppda", "home_deep_completions", "away_deep_completions"
    ]].copy()

    home.columns = [
        "team", "Pts", "goals", "goals_against", "xG", "xGA",
        "ppda", "ppda_allowed", "deep_completions", "deep_allowed"
    ]

    away = df_us[[
        "away_team", "away_points", "away_goals", "home_goals", "away_xg", "home_xg",
        "away_ppda", "home_ppda", "away_deep_completions", "home_deep_completions"
    ]].copy()

    away.columns = home.columns

    # Stack home + away
    df_season = pd.concat([home, away], ignore_index=True)

    # -----------------------------
    # Compute match results
    # -----------------------------
    df_season["W"] = np.where(df_season["Pts"] == 3, 1, 0)
    df_season["D"] = np.where(df_season["Pts"] == 1, 1, 0)
    df_season["L"] = np.where(df_season["Pts"] == 0, 1, 0)

    # -----------------------------
    # Aggregate to season level
    # -----------------------------
    df_us = (
        df_season
        .groupby("team", as_index=False)
        .agg({
            "W": "sum",
            "D": "sum",
            "L": "sum",
            "Pts": "sum",
            "goals": "sum",
            "goals_against": "sum",
            "xG": "sum",
            "xGA": "sum",
            "ppda": "mean",
            "ppda_allowed": "mean",
            "deep_completions": "sum",
            "deep_allowed": "sum"
        })
    )

    # -----------------------------
    # STEP 4: Resolve team name mismatches
    # -----------------------------
    try:
        df_us = apply_name_map(df_us, league=league)
    except Exception as e:
        print("ERROR in apply_name_map:", e)
        raise

    # -----------------------------
    # STEP 5: Merge FBref + Understat
    # -----------------------------
    raw = pd.merge(
        df_fb,
        df_us,
        on="team",
        how="inner",
        suffixes=("_fbref", "_understat")
    )

    # -----------------------------
    # STEP 6: Persist raw features
    # -----------------------------
    base_dir = Path("data/raw")
    league_dir = base_dir / league
    league_dir.mkdir(parents=True, exist_ok=True)

    path = league_dir / f"{league}_{season}.pkl"

    with open(path, "wb") as f:
        pickle.dump(raw, f)

    print(f"{league} {season} Pickle saved!")


def main():
    """
    Run feature generation for all leagues & seasons.
    Designed to fail gracefully per (league, season).
    """
    for league in tqdm(LEAGUES, desc="Leagues"):
        for season in tqdm(SEASONS, desc=f"{league} seasons", leave=False):
            try:
                create_features(season=season, league=league)
            except Exception as e:
                print(f"Failed for {league} {season}")
                print(e)


if __name__ == "__main__":
    main()
