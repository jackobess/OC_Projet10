"""
step3_excel_to_db.py       
Nommé step3 mais independant de step2_indexer.py et peut-etre lancer seul pour recharger la base SQLite à partir de l'Excel source

Charge XLS_FILE (config.py  / feuilles "Données NBA" + "Equipe") dans une base SQLite, avec validation Pydantic ligne par ligne avant insertion.

Usage:
    python step3_excel_to_db.py "inputs/regular NBA.xlsx" --db vector_db/nba.db --season 2024-25

    python step3_excel_to_db.py          => par défaut: charge XLS_FILE dans DB_FILE, saison lue depuis DATA_INFO_FILE (config.ini)
"""

import argparse
import configparser
import sys
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, ForeignKey, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker

from utils.config import DATA_INFO_FILE, DB_FILE, XLS_FILE

Base = declarative_base()

DEFAULT_SEASON = "2024-25"

def read_season_from_ini(ini_path: str = DATA_INFO_FILE) -> str:
    """Lit la saison depuis data_info.ini (section [sources_info]). Fallback sur DEFAULT_SEASON."""
    config = configparser.ConfigParser()
    read_ok = config.read(ini_path, encoding="utf-8")
    if read_ok and config.has_option("sources_info", "saison"):
        saison = config.get("sources_info", "saison")
        print(f"Saison lue depuis {ini_path}: {saison}")
        return saison
    print(f"Pas de saison trouvée dans {ini_path}, fallback sur '{DEFAULT_SEASON}'.")
    return DEFAULT_SEASON

Base = declarative_base()

# ---------------------------------------------------------------------------
# SQLAlchemy models
# ---------------------------------------------------------------------------

class Team(Base):
    __tablename__ = "teams"
    team_code = Column(String(3), primary_key=True)
    full_name = Column(String, nullable=False)


class Player(Base):
    __tablename__ = "players"
    player_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False, unique=True)


class PlayerSeasonStats(Base):
    __tablename__ = "stats"
    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    team_code = Column(String(3), ForeignKey("teams.team_code"), nullable=False)
    season = Column(String, nullable=False)

    age = Column(Integer)
    games_played = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    minutes = Column(Float)
    points = Column(Integer)
    field_goals_made = Column(Integer)
    field_goals_attempted = Column(Integer)
    field_goal_pct = Column(Float)
    three_points_made = Column(Integer)
    three_points_attempted = Column(Integer)
    three_point_pct = Column(Float)
    free_throws_made = Column(Integer)
    free_throws_attempted = Column(Integer)
    free_throw_pct = Column(Float)
    offensive_rebounds = Column(Integer)
    defensive_rebounds = Column(Integer)
    rebounds = Column(Integer)
    assists = Column(Integer)
    turnovers = Column(Integer)
    steals = Column(Integer)
    blocks = Column(Integer)
    personal_fouls = Column(Integer)
    fantasy_points = Column(Integer)
    double_doubles = Column(Integer)
    triple_doubles = Column(Integer)
    plus_minus = Column(Float)
    offensive_rating = Column(Float)
    defensive_rating = Column(Float)
    net_rating = Column(Float)
    assist_pct = Column(Float)
    assist_to_turnover_ratio = Column(Float)
    assist_ratio = Column(Float)
    offensive_rebound_pct = Column(Float)
    defensive_rebound_pct = Column(Float)
    rebound_pct = Column(Float)
    turnover_ratio = Column(Float)
    effective_field_goal_pct = Column(Float)
    true_shooting_pct = Column(Float)
    usage_pct = Column(Float)
    pace = Column(Float)
    player_impact_estimate = Column(Float)
    possessions = Column(Integer)

    __table_args__ = (UniqueConstraint("player_id", "season", name="uq_player_season"),)


class Match(Base):
    """Stub — pas de données match-by-match dans l'Excel source pour l'instant."""
    __tablename__ = "matches"
    match_id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String)
    home_team = Column(String(3), ForeignKey("teams.team_code"))
    away_team = Column(String(3), ForeignKey("teams.team_code"))
    home_score = Column(Integer)
    away_score = Column(Integer)


class Report(Base):
    """Stub — table prévue pour du texte de synthèse (scouting/analyse), pas de source pour l'instant."""
    __tablename__ = "reports"
    report_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=True)
    title = Column(String)
    content = Column(String)
    created_at = Column(String)


# ---------------------------------------------------------------------------
# Pydantic validation
# ---------------------------------------------------------------------------

class PlayerStatRow(BaseModel):
    player: str
    team: str = Field(min_length=3, max_length=3)
    age: int = Field(ge=18, le=50)
    games_played: int = Field(ge=0, le=82)
    wins: int = Field(ge=0)
    losses: int = Field(ge=0)
    minutes: float = Field(ge=0)
    points: int = Field(ge=0)
    field_goals_made: int = Field(ge=0)
    field_goals_attempted: int = Field(ge=0)
    field_goal_pct: float = Field(ge=0, le=100)
    three_points_made: int = Field(ge=0)
    three_points_attempted: int = Field(ge=0)
    three_point_pct: float = Field(ge=0, le=100)
    free_throws_made: int = Field(ge=0)
    free_throws_attempted: int = Field(ge=0)
    free_throw_pct: float = Field(ge=0, le=100)
    offensive_rebounds: int = Field(ge=0)
    defensive_rebounds: int = Field(ge=0)
    rebounds: int = Field(ge=0)
    assists: int = Field(ge=0)
    turnovers: int = Field(ge=0)
    steals: int = Field(ge=0)
    blocks: int = Field(ge=0)
    personal_fouls: int = Field(ge=0)
    fantasy_points: int
    double_doubles: int = Field(ge=0)
    triple_doubles: int = Field(ge=0)
    plus_minus: float
    offensive_rating: float
    defensive_rating: float
    net_rating: float
    assist_pct: float
    assist_to_turnover_ratio: float
    assist_ratio: float
    offensive_rebound_pct: float
    defensive_rebound_pct: float
    rebound_pct: float
    turnover_ratio: float
    effective_field_goal_pct: float
    true_shooting_pct: float
    usage_pct: float
    pace: float
    player_impact_estimate: float
    possessions: int = Field(ge=0)

    @field_validator("team")
    @classmethod
    def team_upper(cls, v: str) -> str:
        return v.strip().upper()


class TeamRow(BaseModel):
    code: str = Field(min_length=3, max_length=3)
    full_name: str


# ---------------------------------------------------------------------------
# Excel -> DataFrames
# ---------------------------------------------------------------------------

# Mapping colonnes Excel -> noms explicites (champs Pydantic / colonnes SQL).
# La colonne 12 ("#12") est corrompue par Excel : l'en-tête d'origine "3PM"
# (3-point Field Goals Made) a été interprété comme une heure et converti
# en time(15:00:00). On la retag "#12" avant le mapping pour la rediriger
# proprement vers "three_points_made", sans dépendre de son nom d'origine cassé.
COLUMN_MAP = {
    "Player": "player",
    "Team": "team",
    "Age": "age",
    "GP": "games_played",
    "W": "wins",
    "L": "losses",
    "Min": "minutes",
    "PTS": "points",
    "FGM": "field_goals_made",
    "FGA": "field_goals_attempted",
    "FG%": "field_goal_pct",
    "#12": "three_points_made",
    "3PA": "three_points_attempted",
    "3P%": "three_point_pct",
    "FTM": "free_throws_made",
    "FTA": "free_throws_attempted",
    "FT%": "free_throw_pct",
    "OREB": "offensive_rebounds",
    "DREB": "defensive_rebounds",
    "REB": "rebounds",
    "AST": "assists",
    "TOV": "turnovers",
    "STL": "steals",
    "BLK": "blocks",
    "PF": "personal_fouls",
    "FP": "fantasy_points",
    "DD2": "double_doubles",
    "TD3": "triple_doubles",
    "+/-": "plus_minus",
    "OFFRTG": "offensive_rating",
    "DEFRTG": "defensive_rating",
    "NETRTG": "net_rating",
    "AST%": "assist_pct",
    "AST/TO": "assist_to_turnover_ratio",
    "AST RATIO": "assist_ratio",
    "OREB%": "offensive_rebound_pct",
    "DREB%": "defensive_rebound_pct",
    "REB%": "rebound_pct",
    "TO RATIO": "turnover_ratio",
    "EFG%": "effective_field_goal_pct",
    "TS%": "true_shooting_pct",
    "USG%": "usage_pct",
    "PACE": "pace",
    "PIE": "player_impact_estimate",
    "POSS": "possessions",
}


def load_players_sheet(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Données NBA", header=1)
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

    cols = list(df.columns)
    cols[11] = "#12"  # colonne 12 (index 11) corrompue par Excel
    df.columns = cols

    df = df.rename(columns=COLUMN_MAP)
    return df


def load_teams_sheet(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Equipe")
    df = df.rename(columns={"Code": "code", "Nom complet de l'équipe": "full_name"})
    return df

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--excel_path", default=XLS_FILE, help="Chemin vers le fichier Excel à charger")
    parser.add_argument("--db", default=DB_FILE, help="Chemin vers la base de données SQLite")
    parser.add_argument("--season", default=None,
                         help="Saison à taguer (défaut: lue depuis inputs/data_info.ini, sinon '2024-25')")
    parser.add_argument("--data-info", default=DATA_INFO_FILE,
                         help="Chemin du fichier data_info.ini")
    args = parser.parse_args()

    season = args.season or read_season_from_ini(args.data_info)

    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{args.db}")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # --- Teams --- (inchangé)
    teams_df = load_teams_sheet(args.excel_path)
    n_teams_ok = 0
    for _, row in teams_df.iterrows():
        try:
            validated = TeamRow(code=row["code"], full_name=row["full_name"])
        except ValidationError as e:
            print(f"[SKIP team] {row.to_dict()} -> {e}", file=sys.stderr)
            continue
        if not session.get(Team, validated.code):
            session.add(Team(team_code=validated.code, full_name=validated.full_name))
        n_teams_ok += 1
    session.commit()
    print(f"Teams chargées : {n_teams_ok}/{len(teams_df)}")

    # --- Players + Stats ---
    n_deleted = (
        session.query(PlayerSeasonStats)
        .filter_by(season=season)
        .delete(synchronize_session=False)
    )
    session.commit()
    if n_deleted:
        print(f"Anciennes stats {season} supprimées : {n_deleted}")

    stats_df = load_players_sheet(args.excel_path)
    n_ok, n_ko = 0, 0
    for _, row in stats_df.iterrows():
        try:
            validated = PlayerStatRow(**row.to_dict())
        except ValidationError as e:
            n_ko += 1
            print(f"[SKIP stat row] player={row.get('player')} -> {e}", file=sys.stderr)
            continue

        player = session.query(Player).filter_by(name=validated.player).one_or_none()
        if player is None:
            player = Player(name=validated.player)
            session.add(player)
            session.flush()

        stat = PlayerSeasonStats(
            player_id=player.player_id,
            team_code=validated.team,
            season=season,
            **validated.model_dump(exclude={"player", "team"}),
        )
        session.add(stat)
        n_ok += 1

    session.commit()
    session.close()
    print(f"Stats chargées : {n_ok} OK / {n_ko} rejetées (saison={season})")


if __name__ == "__main__":
    main()