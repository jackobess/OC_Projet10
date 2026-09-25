# prepare_data.py         => python prepare_data.py
# 1ere étape du pipeline de preparation du RAG : extraction texte -> metadata -> nettoyage -> sauvegarde des documents nettoyés (inc. cleaned_documents.pkl)

import argparse
import configparser
import logging
import pickle
from pathlib import Path
from typing import Optional

from utils.config import INPUT_DIR, DATA_INFO_FILE, VECTOR_DB_DIR
from utils.data_loader import download_and_extract_zip, load_and_parse_files
from utils.cleaner import clean_documents

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

CLEANED_DOCS_FILENAME = "cleaned_documents.pkl"

def load_data_info(path: str = DATA_INFO_FILE) -> dict:
    """Lit data_info.ini (section [sources_info]) s'il existe. Retourne un dict vide sinon."""
    config = configparser.ConfigParser()
    read_ok = config.read(path, encoding="utf-8")
    if not read_ok:
        logging.info(f"Pas de {path} trouvé, aucune info supplémentaire injectée dans les metadata.")
        return {}
    if "sources_info" not in config:
        logging.warning(f"{path} trouvé mais pas de section [sources_info].")
        return {}
    return dict(config["sources_info"])


def apply_data_info_to_metadata(raw_documents, data_info: dict):
    """Injecte les clés de data_info.ini (ex: saison) dans metadata de chaque RawDocument."""
    if not data_info:
        return
    for doc in raw_documents:
        for key, value in data_info.items():
            setattr(doc.metadata, key, value)
    logging.info(f"Metadata enrichies avec: {data_info} ({len(raw_documents)} documents).")


def save_cleaned_documents(cleaned_documents, output_dir: str = VECTOR_DB_DIR) -> Path:
    """Sérialise la liste des documents nettoyés (1 élément = 1 fichier source) sous forme de pickle."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    pkl_path = output_path / CLEANED_DOCS_FILENAME
    with open(pkl_path, "wb") as f:
        pickle.dump(cleaned_documents, f)
    logging.info(f"Documents nettoyés sauvegardés: {pkl_path} ({len(cleaned_documents)} documents).")
    return pkl_path


def run_preparation(input_directory: str, data_url: Optional[str] = None) -> Optional[Path]:
    """Exécution du pipeline de préparation : extraction -> parsing -> metadata -> nettoyage -> sauvegarde."""

    logging.info("--- Démarrage du processus de préparation de la donnée ---")

    # --- Étape 1: Téléchargement et Extraction (Optionnel) ---
    if data_url:
        logging.info(f"Tentative de téléchargement depuis l'URL: {data_url}")
        success = download_and_extract_zip(data_url, input_directory)
        if not success:
            logging.error("Échec du téléchargement ou de l'extraction. Arrêt.")
            return None
    else:
        logging.info(f"Aucune URL fournie. Utilisation des fichiers locaux dans: {input_directory}")

    # --- Étape 2: Chargement et Parsing des Fichiers ---
    logging.info(f"Chargement et parsing des fichiers depuis: {input_directory}")
    raw_documents = load_and_parse_files(input_directory)

    if not raw_documents:
        logging.warning("Aucun document valide n'a été chargé. Vérifiez le contenu du dossier d'entrée.")
        logging.info("--- Processus de préparation terminé (aucun document traité) ---")
        return None

    # --- Étape 2bis: Injection des infos de data_info.ini (saison, etc.) dans les metadata ---
    data_info = load_data_info()
    apply_data_info_to_metadata(raw_documents, data_info)

    # --- Étape 2ter: Nettoyage des documents (Pydantic AI) ---
    logging.info("Nettoyage des documents via Pydantic AI (Mistral)...")
    cleaned_documents = clean_documents(raw_documents)
    if not cleaned_documents:
        logging.warning("Aucun document nettoyé. Arrêt.")
        return None

    # --- Étape 3: Sauvegarde des documents nettoyés (consommés ensuite par indexer.py) ---
    pkl_path = save_cleaned_documents(cleaned_documents)

    logging.info("--- Processus de préparation terminé avec succès ---")
    logging.info(f"Nombre de documents bruts traités: {len(raw_documents)}")
    logging.info(f"Nombre de documents nettoyés: {len(cleaned_documents)}")

    return pkl_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script de préparation de la donnée pour le système RAG")
    parser.add_argument("--input-dir", type=str, default=INPUT_DIR,
                         help=f"Répertoire contenant les fichiers sources (par défaut: {INPUT_DIR})")
    parser.add_argument("--data-url", type=str, default=None,
                         help="URL optionnelle pour télécharger et extraire un fichier inputs.zip")
    args = parser.parse_args()

    run_preparation(input_directory=args.input_dir, data_url=args.data_url)