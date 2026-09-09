# ragas_dataset.py
"""
Jeu de test catégorisé pour l'évaluation RAGAS du prototype RAG.

Catégories :
- "simple"      : factuelle directe, réponse attendue dans un seul chunk Reddit
- "complexe"    : nécessite de croiser plusieurs commentaires/threads
- "chiffree"    : agrégation calculable sur regular_NBA.xlsx (ground truth calculé via pandas)
- "hors_scope"  : question dont la réponse n'existe PAS dans les données disponibles
                   (granularité par match / domicile-extérieur absente du xlsx)
- "bruitee"     : cible un document où du bruit (pub/OCR) a été repéré dans les chunks

IMPORTANT :
- Les questions "chiffree" ont un ground_truth réel, calculé depuis regular_NBA.xlsx
  (feuille "Données NBA", header=1) le 2026-09-07.
- Les questions "simple"/"complexe"/"bruitee" ont un ground_truth écrit à partir de la
  lecture réelle des PDF Reddit 1-4.
"""

RAGAS_DATASET = [
    # --- Chiffrées : ground truth calculé depuis regular_NBA.xlsx ---
    {
        "question": "Quel joueur a le meilleur pourcentage de réussite à 3 points sur la saison, "
                    "parmi ceux ayant tenté au moins 100 tirs à 3 points ?",
        "ground_truth": "Seth Curry (CHA), avec 45.6% de réussite à 3 points sur 184 tentatives.",
        "category": "chiffree",
    },
    {
        "question": "Quel joueur a marqué le plus de points sur la saison régulière ?",
        "ground_truth": "Shai Gilgeous-Alexander (OKC), avec 2485 points sur la saison.",
        "category": "chiffree",
    },
    {
        "question": "Quel joueur a pris le plus de rebonds sur la saison ?",
        "ground_truth": "Ivica Zubac (LAC), avec 1008 rebonds sur la saison.",
        "category": "chiffree",
    },
    {
        "question": "Quel joueur a le plus de passes décisives sur la saison ?",
        "ground_truth": "Trae Young (ATL), avec 882 passes décisives sur la saison.",
        "category": "chiffree",
    },

    # --- Hors-scope : la donnée n'existe pas, ground truth = absence d'info ---
    {
        "question": "Quel joueur a le meilleur pourcentage de réussite à 3 points sur les 5 derniers matchs ?",
        "ground_truth": "Information non disponible : les données ne contiennent que des statistiques "
                         "agrégées sur la saison entière, sans détail match par match.",
        "category": "hors_scope",
    },
    {
        "question": "Compare les statistiques de rebonds de l'équipe à domicile et à l'extérieur.",
        "ground_truth": "Information non disponible : les données ne distinguent pas les matchs "
                         "joués à domicile de ceux joués à l'extérieur.",
        "category": "hors_scope",
    },

    # --- Simples (Reddit) : réponse dans le post d'origine ou un commentaire très upvoté ---
    {
        "question": "Dans le thread Reddit 'Who are teams in the playoffs that have impressed you?', "
                    "quelles équipes reviennent le plus souvent citées comme ayant impressionné les fans ?",
        "ground_truth": "Le post d'origine cite le Magic. Dans les commentaires les plus upvotés, "
                        "les Pacers (Indiana) et les Timberwolves (Minnesota) sont cités ensemble "
                        "comme les plus impressionnants, les Wolves étant aussi cités seuls, et les "
                        "Pistons (Detroit) sont également mentionnés comme surprenants.",
        "category": "simple",
    },

    # --- Complexes (Reddit) : nécessite de croiser un thread avec une connaissance NBA générale ---
    {
        "question": "D'après le thread Reddit sur l'avantage du terrain en playoffs, une équipe a-t-elle "
                    "déjà atteint les Finals NBA sans jamais avoir eu l'avantage du terrain lors des tours "
                    "précédents (contrairement aux Oilers en NHL cette année-là) ?",
        "ground_truth": "Non. Selon les commentaires du thread, ce scénario ne s'est jamais produit en NBA. "
                        "Le cas le plus proche reste les Houston Rockets de 1995, qui ont remporté le titre "
                        "sans avoir l'avantage du terrain à aucun moment des playoffs, mais aucune équipe "
                        "n'a atteint les Finals dans cette configuration exacte (contrairement à la NHL, "
                        "où les Oilers l'ont fait la même année selon le post).",
        "category": "complexe",
    },

    # --- Bruitée : cible le chunk où le post Reddit est mélangé à la pub Xometry (voir AUDIT_REPORT.md §5) ---
    {
        "question": "Selon l'auteur du post 'Who are teams in the playoffs that have impressed you?', "
                    "quel duo de jeunes ailiers du Magic cite-t-il comme son duo préféré de la ligue ?",
        "ground_truth": "Paolo (Banchero) et Franz (Wagner), cités par l'auteur du post comme son duo "
                        "d'ailiers jeunes préféré de la ligue, pour leur adresse au tir et leur activité "
                        "défensive malgré les difficultés au tir du Magic dans l'ensemble.",
        "category": "bruitee",
    },
]
