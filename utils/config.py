# utils/config.py
# Configuration globale du projet : chemins, clés API, modèles, paramètres de chunking, etc.

import os
from dotenv import load_dotenv

# Charger les variables d'environnement du fichier .env
load_dotenv()

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")
    # Vous pouvez choisir de lever une exception ici ou de continuer avec des fonctionnalités limitées

# --- Modèles Mistral ---
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
EMBEDDING_MODEL = "mistral-embed"
CLEANING_MODEL = "mistral-small-latest" # Modèle pour le nettoyage de texte
MODEL_NAME = "mistral-small-latest"     # Ou un autre modèle comme mistral-large-latest
SQL_MODEL_NAME = "mistral-large-latest" # sql_tool — routing + génération SQL
RAGAS_JUDGE_MODEL = "mistral-large-latest" # juge RAGAS

# --- Configuration de la preparation des données et de l'Indexation ---
# INPUT_DATA_URL = os.getenv("INPUT_DATA_URL") # Décommentez si vous utilisez une URL
INPUT_DIR = "inputs"                    # Dossier pour les données sources
INPUT_TXT_DIR = "inputs_txt"            # Dossier pour les données sources après extraction
VECTOR_DB_DIR = "vector_db"   # Dossier pour stocker l'index Faiss, les chunks et la base de données SQLite
# VECTOR_DB_DIR = "vector_db_bigchunks"   # Dossier VECTOR_DB_DIR version bigchunks ou smallchunks (testing)
FAISS_INDEX_FILE = os.path.join(VECTOR_DB_DIR, "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = os.path.join(VECTOR_DB_DIR, "document_chunks.pkl")
DB_FILE = os.path.join(VECTOR_DB_DIR, "nba.db")  # Database from source excel pour sql_tool
DB_URI = f"sqlite:///{DB_FILE}"
DATA_INFO_FILE = os.path.join(INPUT_DIR, "data_info.ini")   # Season info, etc.
XLS_FILE = os.path.join(INPUT_DIR, "regular NBA.xlsx")  # source Excel pour sql_tool (players, teams, stats)

CHUNK_SIZE = 1000                       # Taille des chunks en *caractères*
CHUNK_OVERLAP = 300                     # Chevauchement en *caractères*
EMBEDDING_BATCH_SIZE = 32               # Taille des lots pour l'API d'embedding

# --- Configuration de la Recherche ---
SEARCH_K = 5                           # Nombre de chunks à récupérer par défaut

# --- Configuration de la Base de Données --- (non utilisé)
DATABASE_DIR = "database"
DATABASE_FILE = os.path.join(DATABASE_DIR, "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}" # URL pour SQLAlchemy

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA" # Nom à personnaliser dans l'interface