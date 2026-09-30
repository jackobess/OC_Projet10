# ragas_dataset.py
"""
Jeu de test catégorisé pour l'évaluation RAGAS du prototype RAG.

Catégories :
- "simple"      : factuelle directe, réponse attendue dans un seul chunk Reddit
- "complexe"    : nécessite de croiser plusieurs commentaires/threads
- "chiffree"    : agrégation calculable sur regular_NBA.xlsx
- "hors_scope"  : question dont la réponse n'existe PAS dans les données disponibles
                   (ex: granularité par match, stats domicile-extérieur absentes du xlsx)
- "bruitee"     : cible un document où du bruit (publicité, ...) a été repér

IMPORTANT :
- Les questions "chiffree" ont un ground_truth réel, calculé depuis regular_NBA.xlsx
- Les questions "simple"/"complexe"/"bruitee" ont un ground_truth écrit à partir de la
  lecture réelle des PDF Reddit 1-4.
"""

RAGAS_DATASET = [
    # --- Chiffrées : ground truth calculé depuis regular_NBA.xlsx ---
    {
        "question": "Quel joueur a le meilleur pourcentage de réussite à 3 points sur la saison, "
                    "parmi ceux ayant tenté au moins 100 tirs à 3 points ?",
        "ground_truth": "Seth Curry, avec 45.6% de réussite.",
        "category": "chiffrée",
    },
    {
        "question": "Quel joueur a marqué le plus de points sur la saison régulière ?",
        "ground_truth": "Shai Gilgeous-Alexander, avec 2485 points sur la saison.",
        "category": "chiffrée",
    },
    {
        "question": "Quel joueur a pris le plus de rebonds sur la saison ?",
        "ground_truth": "Ivica Zubac, avec 1008 rebonds sur la saison.",
        "category": "chiffrée",
    },
    {
        "question": "Quels sont les 3 meilleurs passeurs décisifs de la saison ?",
        "ground_truth": "Trae Young, avec 882 passes décisives, suivi de Nikola Jokic avec 714 et de James Harden avec 687.",
        "category": "chiffrée",
    },

    # --- Hors-scope : la donnée n'existe pas dans les données disponibles ---
    {
        "question": "Quel joueur a le meilleur pourcentage de réussite à 3 points sur les 5 derniers matchs ?",
        "ground_truth": "Désolé, les infos ne sont pas disponibles dans la base de connaissances de ce RAG.",
        "category": "hors_scope",
    },
    {
        "question": "Compare les statistiques de rebonds de l'équipe à domicile et à l'extérieur.",
        "ground_truth": "Désolé, les infos ne sont pas disponibles dans la base de connaissances de ce RAG.",
        "category": "hors_scope",
    },
    {
        "question": "Quelle meteo pour demain à Nantes?",
        "ground_truth": "Désolé, je ne peux répondre qu'à des questions concernant la NBA.",
        "category": "hors_scope",
    },

    # --- Simple : réponse dans un seul post ---
    {
        "question": "Selon l'auteur du post 'Who are teams in the playoffs that have impressed you?', "
                    "quel duo de jeunes ailiers du Magic cite-t-il comme son duo préféré de la ligue ?",
        "ground_truth": "Paolo Banchero et Franz Wagner.",
        "category": "simple",
    },
       
    # --- Complexe (Reddit) : réponse dans plusieurs posts ---
    {
        "question": "Quelles équipes ont le plus impressionné lors des playoffs ?",
        "ground_truth": "D'apres les commentaires, les équipes les plus impressionnants sont les Wolwes (Minnesota), les Indiana Pacers et les Pistons (Detroit).",
        "category": "complexe",
    },
    
    # --- Bruitée (Reddit) : présence de publicité, saut de page, etc autour des posts concernés ---
    {
        "question": "En quoi les fans considerent Randle comme une révélation cette saison?",
        "ground_truth": "Principalement à cause de son impact physique sur le jeu, son bully ball, ses spin moves, son implication défensive et sa meilleure efficacité à 3pts.",
        "category": "bruitée",
    },

]
