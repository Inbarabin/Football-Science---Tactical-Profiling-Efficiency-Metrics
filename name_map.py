import logging
import soccerdata as sd

# Silence noisy loggers from external libraries (e.g. soccerdata)
logging.getLogger().setLevel(logging.ERROR)

"""
Canonical Team Name Mapping

Purpose
-------
Unify team names across different data sources (FBref, Understat),
which often use slightly different naming conventions.

This module ensures consistent team identifiers
throughout the entire data pipeline.
"""

# ============================================================
# PREMIER LEAGUE
# ============================================================
PL_NAME_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester City",
    "Manchester Utd": "Manchester United",
    "Newcastle Utd": "Newcastle United",
    "Sheffield Utd": "Sheffield United",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham",
    "Wolves": "Wolverhampton Wanderers",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Nott'ham Forest": "Nottingham Forest",
    "Luton Town": "Luton",
    "Ipswich Town": "Ipswich"
}

# ============================================================
# LA LIGA
# ============================================================
LA_LIGA_NAME_MAP = {
    "Alavés": "Alaves",
    "Almería": "Almeria",
    "Athletic Club": "Athletic Club",
    "Atlético Madrid": "Atletico Madrid",
    "Barcelona": "Barcelona",
    "Betis": "Real Betis",
    "Celta Vigo": "Celta Vigo",
    "Cádiz": "Cadiz",
    "Eibar": "Eibar",
    "Elche": "Elche",
    "Espanyol": "Espanyol",
    "Getafe": "Getafe",
    "Girona": "Girona",
    "Granada": "Granada",
    "Huesca": "SD Huesca",
    "Las Palmas": "Las Palmas",
    "Leganés": "Leganes",
    "Levante": "Levante",
    "Mallorca": "Mallorca",
    "Osasuna": "Osasuna",
    "Rayo Vallecano": "Rayo Vallecano",
    "Real Madrid": "Real Madrid",
    "Real Sociedad": "Real Sociedad",
    "Valladolid": "Real Valladolid",
    "Villarreal": "Villarreal"
}

# ============================================================
# SERIE A
# ============================================================
SERIE_A_NAME_MAP = {
    "Atalanta": "Atalanta",
    "Benevento": "Benevento",
    "Bologna": "Bologna",
    "Cagliari": "Cagliari",
    "Como": "Como",
    "Cremonese": "Cremonese",
    "Crotone": "Crotone",
    "Empoli": "Empoli",
    "Fiorentina": "Fiorentina",
    "Frosinone": "Frosinone",
    "Genoa": "Genoa",
    "Hellas Verona": "Verona",
    "Inter": "Inter",
    "Juventus": "Juventus",
    "Lazio": "Lazio",
    "Lecce": "Lecce",
    "Milan": "AC Milan",
    "Monza": "Monza",
    "Napoli": "Napoli",
    "Parma": "Parma Calcio 1913",
    "Roma": "Roma",
    "Salernitana": "Salernitana",
    "Sampdoria": "Sampdoria",
    "Sassuolo": "Sassuolo",
    "Spezia": "Spezia",
    "Torino": "Torino",
    "Udinese": "Udinese",
    "Venezia": "Venezia"
}

# ============================================================
# BUNDESLIGA
# ============================================================
BUNDESLIGA_NAME_MAP = {
    "Arminia": "Arminia Bielefeld",
    "Augsburg": "Augsburg",
    "Bayern Munich": "Bayern Munich",
    "Bochum": "Bochum",
    "Darmstadt 98": "Darmstadt",
    "Dortmund": "Borussia Dortmund",
    "Eint Frankfurt": "Eintracht Frankfurt",
    "Freiburg": "Freiburg",
    "Gladbach": "Borussia M.Gladbach",
    "Heidenheim": "FC Heidenheim",
    "Hertha BSC": "Hertha Berlin",
    "Hoffenheim": "Hoffenheim",
    "Köln": "FC Cologne",
    "Leverkusen": "Bayer Leverkusen",
    "Mainz 05": "Mainz 05",
    "RB Leipzig": "RasenBallsport Leipzig",
    "Schalke 04": "Schalke 04",
    "St. Pauli": "St. Pauli",
    "Stuttgart": "VfB Stuttgart",
    "Union Berlin": "Union Berlin",
    "Werder Bremen": "Werder Bremen",
    "Wolfsburg": "Wolfsburg",
    "Holstein Kiel": "Holstein Kiel"
}

# ============================================================
# LIGUE 1
# ============================================================
LIGUE_1_NAME_MAP = {
    "Angers": "Angers",
    "Auxerre": "Auxerre",
    "Bordeaux": "Bordeaux",
    "Brest": "Brest",
    "Clermont Foot": "Clermont Foot",
    "Dijon": "Dijon",
    "Lens": "Lens",
    "Le Havre": "Le Havre",
    "Lille": "Lille",
    "Lorient": "Lorient",
    "Lyon": "Lyon",
    "Marseille": "Marseille",
    "Metz": "Metz",
    "Monaco": "Monaco",
    "Montpellier": "Montpellier",
    "Nantes": "Nantes",
    "Nice": "Nice",
    "Nîmes": "Nimes",
    "Paris S-G": "Paris Saint Germain",
    "Reims": "Reims",
    "Rennes": "Rennes",
    "Saint-Étienne": "Saint-Etienne",
    "Strasbourg": "Strasbourg",
    "Toulouse": "Toulouse",
    "Troyes": "Troyes"
}

# ============================================================
# MASTER LEAGUE MAP
# ============================================================
LEAGUE_NAME_MAP = {
    "ENG-Premier League": PL_NAME_MAP,
    "ESP-La Liga": LA_LIGA_NAME_MAP,
    "ITA-Serie A": SERIE_A_NAME_MAP,
    "GER-Bundesliga": BUNDESLIGA_NAME_MAP,
    "FRA-Ligue 1": LIGUE_1_NAME_MAP,
}


def apply_name_map(df, league, col="team"):
    """
    Apply canonical team name mapping to a DataFrame column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing team names
    league : str
        League identifier (must exist in LEAGUE_NAME_MAP)
    col : str, default="team"
        Column containing team names to be normalized

    Returns
    -------
    pandas.Series
        Series with unified team names

    Raises
    ------
    ValueError
        If league is not supported
    """

    try:
        name_map = LEAGUE_NAME_MAP[league]
    except KeyError:
        raise ValueError(f"Unsupported league: {league}")

    # NOTE:
    # .replace keeps original value if key is missing,
    # which is safer than hard-failing on unseen teams
    return df[col].replace(name_map)


def import_names(leagues="Big 5 European Leagues Combined", seasons=2025):
    fb = sd.FBref(leagues=leagues, seasons=seasons)
    us = sd.Understat(leagues=leagues, seasons=seasons)

    df_fb = fb.read_team_season_stats(stat_type="standard")
    print("\nFBref team names (from index):\n")
    print(sorted(df_fb.index.tolist()))

    df_us = us.read_team_match_stats()
    print("\nUnderstat team names (from index):\n")
    print(sorted(df_us["home_team"].unique()))


def main():


    #for names in ["ENG-Premier League", "ESP-La Liga", "FRA-Ligue 1", "GER-Bundesliga", "ITA-Serie A"]:
     #   for i in range(2020, 2025):
      #      print(f"\n{i}:")
       #     import_names(leagues=names, seasons=i)
    fb = sd.FBref(leagues="ENG-Premier League", seasons=2025)
    df_fb = fb.read_team_season_stats(stat_type="standard")
    print("\nFBref team names (from index):\n")
    print(sorted(df_fb.index.tolist()))
    print(df_fb.columns)

    df_fb_flat = df_fb.copy()
    df_fb_flat.columns = ["_".join(col) for col in df_fb.columns]
    print(df_fb_flat["Playing Time_MP"])
    print(df_fb_flat["url_"])







if __name__ == "__main__":
    main()

