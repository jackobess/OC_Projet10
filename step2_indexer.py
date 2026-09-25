# step2_indexer.py         => python step2_indexer.py
# peut être lancé seul pour recharger l'index Faiss à partir des documents nettoyés (cleaned_documents.pkl) produits par step1_prepare_data.py

import argparse
import logging
import pickle
from pathlib import Path

from utils.vector_store import VectorStoreManager
from utils.config import VECTOR_DB_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CLEANED_DOCS_FILENAME = "cleaned_documents.pkl"

def load_cleaned_documents(input_dir: str = VECTOR_DB_DIR):
    """Recharge la liste des documents nettoyés produite par step1_prepare_data.py."""
    pkl_path = Path(input_dir) / CLEANED_DOCS_FILENAME
    if not pkl_path.exists():
        logging.error(
            f"Fichier introuvable: {pkl_path}. "
            f"Lance d'abord 'python step1_prepare_data.py' pour générer les documents nettoyés."
        )
        return None
    with open(pkl_path, "rb") as f:
        cleaned_documents = pickle.load(f)
    logging.info(f"Documents nettoyés rechargés: {pkl_path} ({len(cleaned_documents)} documents).")
    return cleaned_documents


def run_indexing(input_dir: str = VECTOR_DB_DIR):
    """Exécution du pipeline de mise en place du système RAG : chargement -> chunking -> embedding -> Faiss."""

    logging.info("--- Démarrage du processus d'indexation ---")

    cleaned_documents = load_cleaned_documents(input_dir)
    if not cleaned_documents:
        logging.info("--- Processus d'indexation terminé (aucun document à indexer) ---")
        return

    # --- Étape: Création/Mise à jour de l'index Vectoriel ---
    logging.info("Initialisation du gestionnaire de Vector Store...")
    vector_store = VectorStoreManager()

    logging.info("Construction de l'index Faiss (chunking, embeddings, validation)...")
    vector_store.build_index(cleaned_documents)

    logging.info("--- Processus d'indexation terminé avec succès ---")
    logging.info(f"Nombre de documents nettoyés traités: {len(cleaned_documents)}")
    if vector_store.index:
        logging.info(f"Nombre de chunks indexés: {vector_store.index.ntotal}")
    else:
        logging.warning("L'index final n'a pas pu être créé ou est vide.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'indexation pour l'application RAG")
    parser.add_argument("--input-dir", type=str, default=VECTOR_DB_DIR,
                         help=f"Répertoire contenant cleaned_documents.pkl (par défaut: {VECTOR_DB_DIR})")
    args = parser.parse_args()

    run_indexing(input_dir=args.input_dir)
