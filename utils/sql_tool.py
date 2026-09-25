"""
sql_tool.py
Tool LangChain : NL -> SQL -> exécution -> résultat brut, pour l'agent NBA Analyst AI.

testing: python -c "from utils.sql_tool import nl_to_sql_and_run; print(nl_to_sql_and_run('< LA QUESTION DU USER C EST ICI !! >'))"
"""

import re
from langchain_community.utilities import SQLDatabase
from langchain_mistralai import ChatMistralAI
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.tools import Tool
from utils.config import SQL_MODEL_NAME, DB_URI

# ---------------------------------------------------------------------
# 0. Saison courante (fixée à l'exécution par MistralChat.py via set_season)
# ---------------------------------------------------------------------
DEFAULT_SEASON = "2024-25"
_current_season = DEFAULT_SEASON

def set_season(season: str) -> None:
    global _current_season
    _current_season = season

def get_season() -> str:
    return _current_season

def has_season_data(season: str) -> bool:
    """Vérifie si la table stats contient au moins une ligne pour cette saison."""
    with db._engine.connect() as conn:
        result = conn.exec_driver_sql(
            "SELECT COUNT(*) FROM stats WHERE season = ?", (season,)
        ).scalar()
    return bool(result)

# ---------------------------------------------------------------------
# 1. Connexion à la base — schéma limité à players/teams/stats
# ---------------------------------------------------------------------
db = SQLDatabase.from_uri(DB_URI, include_tables=["players", "teams", "stats"])

# ---------------------------------------------------------------------
# 2. Few-shot examples (question -> SQL)
#    __SEASON__ est un placeholder remplacé par la saison courante au
#    moment de construire le prompt (cf. nl_to_sql_and_run).
# ---------------------------------------------------------------------
examples = [
    {
        "question": "Quel joueur a le meilleur pourcentage de réussite à 3 points sur la saison parmi ceux ayant tenté au moins 100 tirs à 3 points ?",
        "query": """SELECT p.name, s.three_point_pct
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.season = '__SEASON__' AND s.three_points_attempted >= 100
ORDER BY s.three_point_pct DESC LIMIT 1;""",
    },
    {
        "question": "Quel joueur a marqué le plus de points sur la saison régulière ?",
        "query": """SELECT p.name, s.points
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.season = '__SEASON__'
ORDER BY s.points DESC LIMIT 1;""",
    },
    {
        "question": "Quel joueur a pris le plus de rebonds sur la saison ?",
        "query": """SELECT p.name, s.rebounds
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.season = '__SEASON__'
ORDER BY s.rebounds DESC LIMIT 1;""",
    },
    {
        "question": "Quel joueur a le plus de passes décisives sur la saison ?",
        "query": """SELECT p.name, s.assists
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.season = '__SEASON__'
ORDER BY s.assists DESC LIMIT 1;""",
    },
    {
        # Pas de colonne "2PM" dans le schéma : paniers à 2pts = FGM total - 3PM
        "question": "Quel joueur a marqué le plus de paniers à 2 points ?",
        "query": """SELECT p.name, (s.field_goals_made - s.three_points_made) AS two_points_made
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.season = '__SEASON__'
ORDER BY two_points_made DESC LIMIT 1;""",
    },
]

example_prompt = PromptTemplate(
    input_variables=["question", "query"],
    template="Question: {question}\nSQL: {query}",
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=(
        "Tu es un générateur de requêtes SQLite expert. "
        "Voici le schéma de la base:\n{schema}\n\n"
        "Règles :\n"
        "- Génère UNIQUEMENT une requête SELECT (pas de markdown, pas de texte autour)\n"
        "- Il n'existe PAS de colonne pour les paniers à 2 points : calcule-la avec (field_goals_made - three_points_made)\n"
        "- Filtre toujours sur s.season={season} lorsque la table stats est utilisée\n"
        "- Si tu ne trouves aucune reference à la période concernée dans la question, considère la saison courante ({season})\n"
        "- Si la question porte sur 'sur la saison' sans autre precision, 'cette saison', 'cette année', 'la saison en cours' considères qu'il s'agit de la saison courante (={season})\n"
        "- Si la question porte sur une donnée qui n'existe PAS dans le schéma ci-dessus NE GÉNÈRE AUCUNE requête et réponds UNIQUEMENT par le mot unique: NODATA\n\n"
        "Exemples:"
    ),
    suffix="Question: {input}\nSQL:",
    input_variables=["input", "schema", "season"],
)

# ---------------------------------------------------------------------
# 3. LLM
# ---------------------------------------------------------------------
llm = ChatMistralAI(model=SQL_MODEL_NAME, temperature=0.1)


def _clean_sql(raw: str) -> str:
    raw = re.sub(r"```sql|```", "", raw).strip()
    return raw


def _is_select_only(sql: str) -> bool:
    return bool(re.match(r"^\s*SELECT\b", sql, re.IGNORECASE)) and not re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH)\b", sql, re.IGNORECASE
    )


def nl_to_sql_and_run_old(question: str) -> str:
    schema = db.get_table_info()

    dynamic_examples = [
        {"question": ex["question"], "query": ex["query"].replace("__SEASON__", _current_season)}
        for ex in examples
    ]
    prompt_template = FewShotPromptTemplate(
        examples=dynamic_examples,
        example_prompt=example_prompt,
        prefix=few_shot_prompt.prefix,
        suffix=few_shot_prompt.suffix,
        input_variables=few_shot_prompt.input_variables,
    )

    prompt = prompt_template.format(input=question, schema=schema, season=_current_season)
    sql_query = _clean_sql(llm.invoke(prompt).content)
    print(f"{'-'*80}\nSQL query:\n{sql_query}\n{'-'*80}")
 
    if sql_query.strip().upper() == "NODATA":
        return "#SQLTOOL: Cette question ne peut pas être répondue avec les données de la base."
 
    if not _is_select_only(sql_query):
        return f"#SQLTOOL: Requête refusée (non-SELECT détecté) : {sql_query}"
 
    try:
        result = db.run(sql_query)
    except Exception as e:
        return f"#SQLTOOL: Erreur SQL ({e}). Requête générée: {sql_query}"
 
    if not result or not result.strip() or result.strip() == "[]":
        return (
            f"#SQLTOOL: Requête exécutée: {sql_query}\n"
            "Résultat: AUCUNE LIGNE TROUVÉE — l'entité demandée n'est probablement pas dans la base "
            "(joueur non présent, saison non couverte, etc.). Ne pas interpréter comme une réponse négative/zéro."
        )
 
    return f"Requête exécutée: {sql_query}\nRésultat: {result}"

def nl_to_sql_and_run(question: str) -> str:
    schema = db.get_table_info()

    dynamic_examples = [
        {"question": ex["question"], "query": ex["query"].replace("__SEASON__", _current_season)}
        for ex in examples
    ]
    prompt_template = FewShotPromptTemplate(
        examples=dynamic_examples,
        example_prompt=example_prompt,
        prefix=few_shot_prompt.prefix,
        suffix=few_shot_prompt.suffix,
        input_variables=few_shot_prompt.input_variables,
    )

    prompt = prompt_template.format(input=question, schema=schema, season=_current_season)
    sql_query = _clean_sql(llm.invoke(prompt).content)
    print(f"{'-'*80}\nSQL query:\n{sql_query}\n{'-'*80}")
 
    if sql_query.strip().upper() == "NODATA":
        return "#SQLTOOL: Cette question ne peut pas être répondue avec les données de la base."
 
    if not _is_select_only(sql_query):
        return f"#SQLTOOL: Requête refusée (non-SELECT détecté) : {sql_query}"
 
    try:
        # Exécution directe via l'engine SQLAlchemy pour récupérer les clés
        with db._engine.connect() as conn:
            query_result = conn.exec_driver_sql(sql_query)
            keys = list(query_result.keys())
            rows = query_result.fetchall()

        if not rows:
            return (
                f"#SQLTOOL: Requête exécutée: {sql_query}\n"
                "Résultat: AUCUNE LIGNE TROUVÉE — l'entité demandée n'est probablement pas dans la base "
                "(joueur non présent, saison non couverte, etc.). Ne pas interpréter comme une réponse négative/zéro."
            )

        # Convertit chaque ligne en dictionnaire {colonne: valeur}
        dict_results = [dict(zip(keys, row)) for row in rows]
        
    except Exception as e:
        return f"#SQLTOOL: Erreur SQL ({e}). Requête générée: {sql_query}"

    return f"Requête exécutée: {sql_query}\nRésultat: {dict_results}"

# ---------------------------------------------------------------------
# 4. Export comme Tool LangChain pour l'agent
# ---------------------------------------------------------------------
nba_sql_tool = Tool(
    name="nba_stats_sql",
    func=nl_to_sql_and_run,
    description=(
    """Utilise cet outil pour:
    - Toute question relative aux statistiques de jeu des joueurs ou équipes NBA (ex: nb points, rebonds, contres, passes, tirs, moyennes, pourcentages ...).
    - Ou bien pour une question concernant l'âge d'un ou plusieurs joueurs, ou encore pour savoir qui joue dans quelle équipe.

    Entrée: la question de l'utilisateur en langage naturel."""
    ),
)