import soccerdata as sd
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import dataframe_image as dfi
def create_features(season=2024, league="ENG-Premier League"):
    """
    Build a season-level efficiency table for all teams in a given league and season.
    Combines data from FBref (passing, possession, results) and Understat (xG, xGA, PPDA, Deep).
    Returns a normalized DataFrame ready for K-Means clustering.
    """

    # --- STEP 1: Load raw data from both sources ---
    fb = sd.FBref(leagues=league, seasons=season)
    us = sd.Understat(leagues=league, seasons=season)

    df_us = us.read_team_match_stats()
    print(df_us.columns)




# FBref data: team season-level stats (standard, passing, and possession)
    df_fb = fb.read_team_season_stats(stat_type="standard")
    df_shooting = fb.read_team_season_stats(stat_type="shooting")
    df_passing = fb.read_team_season_stats(stat_type="passing")
    df_possession = fb.read_team_season_stats(stat_type="possession")
    df_defense = fb.read_team_season_stats(stat_type="defense")

    print("\nFBref team names (from index):")
    print(sorted(df_fb.index.tolist()))

    for name, df in zip(["df_fb", "df_shooting", "df_passing", "df_possession", "df_defense"],
                        [df_fb, df_shooting, df_passing, df_possession, df_defense]):
        print(f"{name}:\n{df.columns.tolist()}\n")


    # Flatten each table individually (very important!)
    for df in [df_fb, df_shooting, df_passing, df_possession, df_defense]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Now rename the shooting columns
    df_shooting.rename(columns={"Standard_SoT": "SoT", "Standard_Sh": "shots"}, inplace=True)
    df_passing.rename(columns={"Total_Cmp": "Cmp"}, inplace=True)
    df_possession.rename(columns={"Carries_PrgC": "PrgC", "Carries_CPA": "CPA"}, inplace=True)
    df_defense.rename(columns={"Tackles_Tkl": "tackles", "Tackles_TklW": "tackles_win",
                               "Blocks_Blocks": "blocks"}, inplace=True)


    # Merge FBref data into one table
    df_fb = (
        df_fb.merge(df_shooting, on="team", how="left", suffixes=("", "_shooting"))
        .merge(df_passing, on="team", how="left", suffixes=("", "_pass"))
        .merge(df_possession, on="team", how="left", suffixes=("", "_poss"))
        .merge(df_defense, on="team", how="left", suffixes=("", "_defense"))

    )


    # Understat data: team match-level stats
    df_us = us.read_team_match_stats()
    print(sorted(df_us["home_team"].unique()))

    # Build home stats DataFrame
    home = df_us[[
        "home_team", "home_points", "home_goals", "away_goals", "home_xg", "away_xg",
        "home_ppda", "away_ppda", "home_deep_completions", "away_deep_completions"
    ]].copy()

    home.columns = [
        "team", "Pts", "goals", "goals_against", "xG", "xGA",
        "ppda", "ppda_allowed", "deep_completions", "deep_allowed"
    ]

    # Build away stats DataFrame
    away = df_us[[
        "away_team", "away_points", "away_goals", "home_goals", "away_xg", "home_xg",
        "away_ppda", "home_ppda", "away_deep_completions", "home_deep_completions"
    ]].copy()

    away.columns = [
        "team", "Pts", "goals", "goals_against", "xG", "xGA",
        "ppda", "ppda_allowed", "deep_completions", "deep_allowed"
    ]

    # Merge home + away into one long table (each row = one game for one team)
    df_season = pd.concat([home, away], ignore_index=True)

    #Compute results W / D / L
    df_season["W"] = np.where(df_season["Pts"] == 3, 1, 0)
    df_season["D"] = np.where(df_season["Pts"] == 1, 1, 0)
    df_season["L"] = np.where(df_season["Pts"] == 0, 1, 0)


    # Now aggregate to season totals per team
    df_us = (
        df_season.groupby("team", as_index=False)
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

    for col in df_us.columns:
        values = df_us[col].head(5).tolist()
        print(f"{col}: {values}")

    # --- STEP 3: Merge FBref + Understat ---
    # Fix team name mismatches between the two sources
    PREM_name_map = {
        "Arsenal": "Arsenal",
        "Aston Villa": "Aston Villa",
        "Bournemouth": "Bournemouth",
        "Brentford": "Brentford",
        "Brighton": "Brighton",
        "Burnley": "Burnley",
        "Chelsea": "Chelsea",
        "Crystal Palace": "Crystal Palace",
        "Everton": "Everton",
        "Fulham": "Fulham",
        "Leeds": "Leeds United",
        "Liverpool": "Liverpool",
        "Manchester City": "Manchester City",
        "Manchester United": "Manchester Utd",
        "Newcastle United": "Newcastle Utd",
        "Nottingham Forest": "Nott'ham Forest",
        "Sunderland": "Sunderland",
        "Tottenham": "Tottenham",
        "West Ham": "West Ham",
        "Wolverhampton Wanderers": "Wolves"
    }

    df_us["team"] = df_us["team"].replace(PREM_name_map)
    merged = pd.merge(df_fb, df_us, on="team", how="inner", suffixes=("_fbref", "_understat"))

    # --- STEP 4: Calculate all 20 engineered features ---

    # OUTCOME / PERFORMANCE
    merged["Matches"] = merged["W"] + merged["D"] + merged["L"]                                               # Total matches played
    merged["wins_perc(overall)"] = merged["W"] / merged["Matches"]                                            # Win percentage (success rate)
    merged["att_style(tact)"] = merged["PrgC"] / merged["PrgP_"]                                               # Progressive carries vs progressive passes (attack style)

    # OFFENSIVE FEATURES
    merged["attack_eff(off)"] = merged["goals"] / merged["xG"]                                                # Finishing efficiency (goals vs expected goals)
    merged["shot_conv(off)"] = merged["goals"] / merged["SoT"]                                                # Conversion rate (goals per shot on target)
    merged["shot_eff(off)"] = merged["SoT"] / merged["shots"]                                                 # Shot accuracy (shots on target ratio)
    merged["poss_eff(off)"] = merged["xG"] / merged["Poss_"]                                                  # xG generated per unit of possession (possession efficiency)
    merged["progresive_eff(off)"] = merged["goals"] / merged["PrgP_"]                                         # Goals per progressive pass (passing productivity)
    merged["carries_eff(off)"] = merged["goals"] / merged["CPA"]                                              # Goals per carry into penalty area (carry efficiency)
    merged["xG_balance(tact)"] = merged["xG"] - merged["xGA"]                                                 # Expected goal difference (attacking dominance)
    merged["possession(tact)"] = merged["Poss_"]                                                              # Average possession percentage
    merged["points_eff(tact)"] = merged["Pts"] / merged["xG_balance(tact)"]                                   # Points gained relative to xG difference (performance vs expectation)
    merged["off_tackles_eff(off)"] = merged["Tackles_Att 3rd"] / merged["tackles_win"]                         # Offensive tackle ratio (pressing in final third)
    merged["att_style_eff(off)"] = merged["att_style(tact)"] * merged["poss_eff(off)"]                        # Effectiveness of attack style (style × chance creation per possession)

    # DEFENSIVE FEATURES
    merged["def_eff(def)"] = merged["xGA"] / merged["goals_against"]                                         # Defensive efficiency (actual vs expected goals conceded)
    merged["ppda_eff(def)"] = 1 / merged["ppda"]                                                              # Pressing intensity (higher = more aggressive pressing)
    merged["deep_allowed_per_game(def)"] = (1 / merged["deep_allowed"]) / merged["Matches"]                         # Deep entries allowed per match (defensive compactness)
    merged["tackles_goals_eff(def)"] = merged["tackles"] / merged["goals_against"]                            # Tackles per goal conceded (defensive resistance)
    merged["tackles_eff(def)"] = merged["tackles_win"] / merged["tackles"]                                    # Successful tackles ratio (tackling efficiency)
    merged["deep_eff(tact)"] = merged["deep_completions"] / merged["deep_allowed"]                            # Ratio of offensive to defensive deep completions (territorial control)
    merged["deep_comp_pos_eff(tact)"] = merged["deep_completions"] / merged["possession(tact)"]               # Deep completions per possession (offensive penetration)
    merged["challenge_eff(def)"] = merged["Challenges_Tkl%"]                                                  # Percentage of challenges won (duel success rate)
    merged["box_efficiency(def)"] = merged["Clr_"] / (merged["deep_allowed"] + merged["Tackles_Def 3rd"])      # Defensive box efficiency (clearances per box entry)
    merged["def_style(tact)"] = merged["Int_"] / merged["blocks"]                                             # Defensive style: interceptions vs blocks (proactive vs reactive defending)
    merged["goal_leading_error(def)"] = 1 / merged["Err_"]                                                        # Errors leading to goals (defensive mistakes)



# --- STEP 5: Clean and handle invalid values ---
    merged = merged.replace([float('inf'), -float('inf')], pd.NA).dropna(subset=["attack_eff(off)"])
    merged = merged.fillna(0)

    # --- STEP 6: Normalize all numerical features for clustering ---
    features = [
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
        "goal_leading_error(def)"
    ]

    print(merged["Matches"].unique())


    scaler = StandardScaler()
    merged_scaled = merged.copy()
    merged_scaled[features] = scaler.fit_transform(merged[features])

    return merged_scaled[["team"] + features], merged[["team"] + features], merged



# Run the function and save the result
df_final,df_final_orgVal,df_full = create_features(season=2024)

# Set pandas display options (optional)
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 200)         # Prevent line wrapping

# Print the first rows of the final efficiency table
print("\n✅ Efficiency table (normalized):\n")
print(df_final.head(10))


dfi.export(df_final.head(20), "features_table.png")

with open("features.pkl", "wb") as f:
    pickle.dump(df_final, f)


with open("features_org.pkl", "wb") as f:
    pickle.dump(df_final_orgVal, f)

df_full.to_csv("final_features.csv", index=False, encoding="utf-8-sig")