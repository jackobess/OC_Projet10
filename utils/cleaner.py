# utils/cleaner.py
"""
Étape de nettoyage du pipeline : Pydantic AI + Mistral, document par document.

"""
import logging
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import INPUT_TXT_DIR, MISTRAL_API_KEY, CLEANING_MODEL, MISTRAL_BASE_URL
from .schemas import RawDocument, CleanedText

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class CleaningResult(BaseModel):
    """Sortie structurée attendue du LLM (validée automatiquement par pydantic-ai)."""

    cleaned_content: str = Field(..., description="Texte nettoyé, sans pub ni artefacts OCR")
    removed_noise: bool = Field(..., description="True si du contenu a été retiré")
    notes: Optional[str] = Field(default=None, description="Ce qui a été retiré, en une phrase courte")


_CLEANING_SYSTEM_PROMPT = """Tu es un outil de nettoyage de texte extrait de documents \
(PDF scannés via OCR, threads Reddit, etc.). Ta seule tâche : retirer le bruit, \
SANS jamais reformuler, résumer ou interpréter le contenu utile.

Retire :
- Publicités, bannières, liens promotionnels, mentions sponsorisées
- Artefacts d'OCR évidents (caractères aberrants, doublons de scan, en-têtes/pieds de page scannés)
- Texte d'interface (ex: "upvote", "share", "reply", bannières cookies)

Ne retire JAMAIS :
- Données chiffrées, statistiques, noms de joueurs, scores
- Contenu qui a un sens même informel (commentaires pertinents, etc.)

Important : tu reçois parfois un EXTRAIT d'un document plus long (découpé en blocs), donc
le texte peut commencer ou finir au milieu d'un post/commentaire. En cas de doute sur un
contenu qui semble coupé/incomplet en début ou fin d'extrait, GARDE-le tel quel plutôt que
de le retirer — ne retire que ce que tu identifies clairement et entièrement comme pub ou
artefact OCR.

Si le texte est déjà propre, renvoie-le identique avec removed_noise=False et notes=null.
"""

_agent: Optional[Agent] = None

# Taille max (en caractères) d'un bloc envoyé au LLM de nettoyage. On découpe le texte
# brut AVANT de l'envoyer plutôt que de compter sur un gros max_tokens en sortie :
# sur les gros PDF Reddit, un appel LLM one-shot sur tout le doc tronque parfois la fin
# (le modèle "décroche" sur du contenu répétitif), peu importe la taille du fichier.
CLEANING_BLOCK_SIZE = 6000
CLEANING_BLOCK_OVERLAP = 0

_cleaning_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CLEANING_BLOCK_SIZE,
    chunk_overlap=CLEANING_BLOCK_OVERLAP,
    length_function=len,
)


def _split_for_cleaning(text: str) -> List[str]:
    """Découpe le texte brut en blocs raisonnables avant nettoyage LLM."""
    return _cleaning_splitter.split_text(text)

def _get_agent() -> Agent:
    """Instancie l'agent Pydantic AI une seule fois (lazy singleton)."""
    global _agent
    if _agent is None:
        if not MISTRAL_API_KEY:
            raise RuntimeError("MISTRAL_API_KEY manquante : impossible d'initialiser l'agent de nettoyage.")
        
        # 1. Configuration du provider ciblant Mistral
        provider = OpenAIProvider(base_url=MISTRAL_BASE_URL, api_key=MISTRAL_API_KEY)
        
        # 2. Modèle OpenAI (chat completions) avec le provider personnalisé
        model = OpenAIChatModel(CLEANING_MODEL, provider=provider)
        
        # 3. Agent initialisé avec l'objet modèle
        _agent = Agent(
            model,
            output_type=CleaningResult,
            system_prompt=_CLEANING_SYSTEM_PROMPT,
        )
    return _agent

def save_cleaned_text(cleaned_doc: CleanedText, output_dir: str = INPUT_TXT_DIR) -> None:
    """Sauvegarde le texte nettoyé dans output_dir sous le nom <stem>_<ext>_clean.txt."""
    source_path_str = cleaned_doc.metadata.source
    if not source_path_str:
        return

    source_path = Path(source_path_str)
    stem = source_path.stem
    ext = source_path.suffix.lstrip(".")  # Récupère l'extension sans le point (ex: pdf, docx)

    # Nom au format : NomDuFichier_pdf_clean.txt
    output_filename = f"{stem}_{ext}_clean.txt" if ext else f"{stem}_clean.txt"

    file_path = Path(output_dir) / output_filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(cleaned_doc.cleaned_content, encoding="utf-8")
    logging.info(f"Document nettoyé sauvegardé : {file_path}")

def _clean_block(block_text: str, source: str, block_num: int, total_blocks: int) -> CleaningResult:
    """Nettoie un seul bloc de texte via l'agent. Fallback sur le texte brut du bloc si erreur."""
    agent = _get_agent()
    try:
        result = agent.run_sync(block_text)
        return result.output
    except Exception as e:  # noqa: BLE001 - fallback volontaire, on ne veut pas planter le pipeline
        logging.error(f"Nettoyage LLM échoué pour '{source}' (bloc {block_num}/{total_blocks}): {e}. Fallback texte brut du bloc.")
        return CleaningResult(cleaned_content=block_text, removed_noise=False, notes="fallback: bloc échoué")


def clean_document(raw_doc: RawDocument) -> CleanedText:
    """
    Nettoie un RawDocument en le découpant en blocs (voir CLEANING_BLOCK_SIZE) pour éviter
    qu'un appel LLM one-shot sur un gros texte ne tronque la fin du document. Les blocs
    nettoyés sont recollés dans l'ordre.
    """
    blocks = _split_for_cleaning(raw_doc.page_content)
    total_blocks = len(blocks)

    cleaned_parts: List[str] = []
    any_noise_removed = False
    all_notes: List[str] = []

    for i, block in enumerate(blocks, start=1):
        if total_blocks > 1:
            logging.info(f"  Nettoyage bloc {i}/{total_blocks} de '{raw_doc.metadata.source}'")
        cleaning = _clean_block(block, raw_doc.metadata.source, i, total_blocks)
        cleaned_parts.append(cleaning.cleaned_content)
        any_noise_removed = any_noise_removed or cleaning.removed_noise
        if cleaning.notes:
            all_notes.append(cleaning.notes)

    try:
        return CleanedText(
            cleaned_content="\n".join(cleaned_parts),
            removed_noise=any_noise_removed,
            notes="; ".join(all_notes) if all_notes else None,
            metadata=raw_doc.metadata,
        )
    except ValidationError as e:
        logging.error(f"CleanedText invalide pour '{raw_doc.metadata.source}': {e}. Fallback texte brut complet.")
        return CleanedText(
            cleaned_content=raw_doc.page_content,
            removed_noise=False,
            notes="fallback: CleanedText invalide après nettoyage par blocs",
            metadata=raw_doc.metadata,
        )


_SKIP_CLEANING_EXTENSIONS = {".csv", ".xlsx", ".xls"}  # tabulaire : pas de nettoyage LLM (candidat à la bascule SQL)


def clean_documents(raw_docs: List[RawDocument]) -> List[CleanedText]:
    """Nettoie une liste de RawDocument, document par document (appels séquentiels).

    Les fichiers tabulaires (csv/xlsx/xls) sont zappés : pas de nettoyage LLM, on les
    fait juste transiter tels quels en CleanedText pour la suite du pipeline.
    """
    cleaned: List[CleanedText] = []
    total = len(raw_docs)
    skipped = 0

    for i, doc in enumerate(raw_docs, start=1):
        ext = Path(doc.metadata.filename).suffix.lower()

        if ext in _SKIP_CLEANING_EXTENSIONS:
            skipped += 1
            logging.info(f"[{i}/{total}] Nettoyage zappé (tabulaire, {ext}): {doc.metadata.source}")
            cleaned_doc = CleanedText(
                cleaned_content=doc.page_content,
                removed_noise=False,
                notes="skipped: fichier tabulaire, pas de nettoyage LLM",
                metadata=doc.metadata,
            )
        else:
            logging.info(f"[{i}/{total}] Nettoyage: {doc.metadata.source}")
            cleaned_doc = clean_document(doc)

        cleaned.append(cleaned_doc)
        save_cleaned_text(cleaned_doc)

    logging.info(f"{len(cleaned)}/{total} documents nettoyés ({skipped} zappé(s) car tabulaires).")
    return cleaned
