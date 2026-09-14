# utils/vector_store.py
import os
import pickle
import faiss
import numpy as np
import logging
from typing import List, Dict, Optional
from pydantic import ValidationError
from mistralai.client import MistralClient
from mistralai.exceptions import MistralAPIException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .config import (
    MISTRAL_API_KEY, EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE,
    FAISS_INDEX_FILE, DOCUMENT_CHUNKS_FILE, CHUNK_SIZE, CHUNK_OVERLAP
)
from .schemas import CleanedText, TextChunk, EmbeddedChunk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _extract_title(cleaned_content: str) -> Optional[str]:
    """Récupère la 1ère ligne non vide du texte nettoyé, utilisée comme titre/contexte du thread."""
    for line in cleaned_content.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return None


class VectorStoreManager:
    """Gère la création, le chargement et la recherche dans un index Faiss."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.document_chunks: List[TextChunk] = []
        self.mistral_client = MistralClient(api_key=MISTRAL_API_KEY)
        self._load_index_and_chunks()

    def _load_index_and_chunks(self):
        """Charge l'index Faiss et les chunks (List[TextChunk]) si les fichiers existent."""
        if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(DOCUMENT_CHUNKS_FILE):
            try:
                logging.info(f"Chargement de l'index Faiss depuis {FAISS_INDEX_FILE}...")
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                logging.info(f"Chargement des chunks depuis {DOCUMENT_CHUNKS_FILE}...")
                with open(DOCUMENT_CHUNKS_FILE, 'rb') as f:
                    self.document_chunks = pickle.load(f)
                logging.info(f"Index ({self.index.ntotal} vecteurs) et {len(self.document_chunks)} chunks chargés.")
            except Exception as e:
                logging.error(f"Erreur lors du chargement de l'index/chunks: {e}")
                self.index = None
                self.document_chunks = []
        else:
            logging.warning("Fichiers d'index Faiss ou de chunks non trouvés. L'index est vide.")

    def _split_documents_to_chunks(self, cleaned_texts: List[CleanedText]) -> List[TextChunk]:
        """Découpe les CleanedText (texte nettoyé) en TextChunk validés par Pydantic."""
        logging.info(f"Découpage de {len(cleaned_texts)} documents nettoyés en chunks (taille={CHUNK_SIZE}, chevauchement={CHUNK_OVERLAP})...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            add_start_index=True,
        )

        all_chunks: List[TextChunk] = []
        skipped = 0
        for doc_counter, cleaned in enumerate(cleaned_texts):
            # Le splitter Langchain attend un Document standard -> on lui donne le texte NETTOYÉ
            langchain_doc = Document(
                page_content=cleaned.cleaned_content,
                metadata=cleaned.metadata.model_dump(),
            )
            chunks = text_splitter.split_documents([langchain_doc])
            logging.info(f"  Document '{cleaned.metadata.filename}' découpé en {len(chunks)} chunks.")

            # Titre/question du thread (1ère ligne du texte nettoyé) -> préfixé à chaque chunk
            # pour ne pas perdre le contexte quand le chunk est isolé (ex: threads Reddit en PDF).
            # Limité aux PDF : pour les xlsx (tableaux), la 1ère ligne n'est pas un "titre" pertinent.
            is_pdf_source = cleaned.metadata.filename.lower().endswith(".pdf")
            title = _extract_title(cleaned.cleaned_content) if is_pdf_source else None

            for i, chunk in enumerate(chunks):
                chunk_text = chunk.page_content
                if title and not chunk_text.startswith(title):
                    chunk_text = f"[Contexte: {title}]\n{chunk_text}"
                try:
                    text_chunk = TextChunk(
                        id=f"{doc_counter}_{i}",
                        text=chunk_text,
                        metadata=cleaned.metadata,
                        chunk_id_in_doc=i,
                        start_index=chunk.metadata.get("start_index", -1),
                    )
                    all_chunks.append(text_chunk)
                except ValidationError as e:
                    skipped += 1
                    logging.warning(f"Chunk invalide ignoré ({cleaned.metadata.source}, chunk {i}): {e}")

        if skipped:
            logging.warning(f"{skipped} chunk(s) rejeté(s) par la validation Pydantic.")
        logging.info(f"Total de {len(all_chunks)} chunks valides créés.")
        return all_chunks

    def _generate_embeddings(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        """Génère les embeddings via l'API Mistral et retourne des EmbeddedChunk validés."""
        if not MISTRAL_API_KEY:
            logging.error("Impossible de générer les embeddings: MISTRAL_API_KEY manquante.")
            return []
        if not chunks:
            logging.warning("Aucun chunk fourni pour générer les embeddings.")
            return []

        logging.info(f"Génération des embeddings pour {len(chunks)} chunks (modèle: {EMBEDDING_MODEL})...")
        all_vectors: List[Optional[List[float]]] = [None] * len(chunks)
        total_batches = (len(chunks) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

        for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
            batch_num = (batch_start // EMBEDDING_BATCH_SIZE) + 1
            batch_chunks = chunks[batch_start:batch_start + EMBEDDING_BATCH_SIZE]
            texts_to_embed = [c.text for c in batch_chunks]

            logging.info(f"  Traitement du lot {batch_num}/{total_batches} ({len(texts_to_embed)} chunks)")
            try:
                response = self.mistral_client.embeddings(
                    model=EMBEDDING_MODEL,
                    input=texts_to_embed
                )
                for offset, data in enumerate(response.data):
                    all_vectors[batch_start + offset] = data.embedding
            except MistralAPIException as e:
                logging.error(f"Erreur API Mistral lors de la génération d'embeddings (lot {batch_num}): {e}")
                logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            except Exception as e:
                logging.error(f"Erreur inattendue lors de la génération d'embeddings (lot {batch_num}): {e}")

        # Construction + validation Pydantic des EmbeddedChunk (juste avant Faiss)
        embedded_chunks: List[EmbeddedChunk] = []
        skipped = 0
        for chunk, vector in zip(chunks, all_vectors):
            if vector is None:
                skipped += 1
                continue
            try:
                embedded_chunks.append(EmbeddedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    embedding=vector,
                ))
            except ValidationError as e:
                skipped += 1
                logging.warning(f"EmbeddedChunk invalide ignoré ({chunk.id}): {e}")

        if skipped:
            logging.warning(f"{skipped} chunk(s) sans embedding valide, exclus de l'index.")

        if not embedded_chunks:
            logging.error("Aucun embedding valide n'a pu être généré.")
            return []

        logging.info(f"{len(embedded_chunks)} EmbeddedChunk validés (Pydantic).")
        return embedded_chunks

    def build_index(self, cleaned_texts: List[CleanedText]):
        """Construit l'index Faiss à partir des documents nettoyés."""
        if not cleaned_texts:
            logging.warning("Aucun document nettoyé fourni pour construire l'index.")
            return

        # 1. Chunking (sur texte nettoyé) + validation Pydantic
        chunks = self._split_documents_to_chunks(cleaned_texts)
        if not chunks:
            logging.error("Le découpage n'a produit aucun chunk valide. Impossible de construire l'index.")
            return

        # 2. Embeddings + validation Pydantic (EmbeddedChunk)
        embedded_chunks = self._generate_embeddings(chunks)
        if not embedded_chunks:
            logging.error("Aucun EmbeddedChunk valide. Abandon de la construction de l'index.")
            self.document_chunks = []
            self.index = None
            if os.path.exists(FAISS_INDEX_FILE): os.remove(FAISS_INDEX_FILE)
            if os.path.exists(DOCUMENT_CHUNKS_FILE): os.remove(DOCUMENT_CHUNKS_FILE)
            return

        # 3. Créer l'index Faiss (similarité cosinus via IndexFlatIP sur vecteurs normalisés)
        embeddings_array = np.array([ec.embedding for ec in embedded_chunks], dtype='float32')
        dimension = embeddings_array.shape[1]
        logging.info(f"Création de l'index Faiss (similarité cosinus, dimension {dimension})...")

        faiss.normalize_L2(embeddings_array)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings_array)
        logging.info(f"Index Faiss créé avec {self.index.ntotal} vecteurs.")

        # 4. On garde les TextChunk (sans le vecteur, déjà dans Faiss) pour la recherche,
        #    dans le même ordre que embedded_chunks pour rester aligné avec l'index Faiss.
        chunk_by_id: Dict[str, TextChunk] = {c.id: c for c in chunks}
        self.document_chunks = [chunk_by_id[ec.id] for ec in embedded_chunks]

        self._save_index_and_chunks()

    def _save_index_and_chunks(self):
        """Sauvegarde l'index Faiss et la liste des TextChunk."""
        if self.index is None or not self.document_chunks:
            logging.warning("Tentative de sauvegarde d'un index ou de chunks vides.")
            return

        os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(DOCUMENT_CHUNKS_FILE), exist_ok=True)

        try:
            logging.info(f"Sauvegarde de l'index Faiss dans {FAISS_INDEX_FILE}...")
            faiss.write_index(self.index, FAISS_INDEX_FILE)
            logging.info(f"Sauvegarde des chunks dans {DOCUMENT_CHUNKS_FILE}...")
            with open(DOCUMENT_CHUNKS_FILE, 'wb') as f:
                pickle.dump(self.document_chunks, f)
            logging.info("Index et chunks sauvegardés avec succès.")
        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de l'index/chunks: {e}")

    def search(self, query_text: str, k: int = 5, min_score: float = None) -> List[Dict[str, any]]:
        """
        Recherche les k chunks les plus pertinents pour une requête.
        Retourne des dicts (comme avant) pour rester compatible avec le reste de l'appli.
        """
        if self.index is None or not self.document_chunks:
            logging.warning("Recherche impossible: l'index Faiss n'est pas chargé ou est vide.")
            return []
        if not MISTRAL_API_KEY:
            logging.error("Recherche impossible: MISTRAL_API_KEY manquante pour générer l'embedding de la requête.")
            return []

        logging.info(f"Recherche des {k} chunks les plus pertinents pour: '{query_text}'")
        try:
            response = self.mistral_client.embeddings(
                model=EMBEDDING_MODEL,
                input=[query_text]
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            faiss.normalize_L2(query_embedding)

            search_k = k * 3 if min_score is not None else k
            scores, indices = self.index.search(query_embedding, search_k)

            results = []
            if indices.size > 0:
                for i, idx in enumerate(indices[0]):
                    if 0 <= idx < len(self.document_chunks):
                        chunk = self.document_chunks[idx]  # TextChunk (pydantic)
                        raw_score = float(scores[0][i])
                        similarity = raw_score * 100

                        min_score_percent = min_score * 100 if min_score is not None else 0
                        if min_score is not None and similarity < min_score_percent:
                            logging.debug(f"Document filtré (score {similarity:.2f}% < minimum {min_score_percent:.2f}%)")
                            continue

                        results.append({
                            "score": similarity,
                            "raw_score": raw_score,
                            "text": chunk.text,
                            "metadata": chunk.metadata.model_dump(),
                        })
                    else:
                        logging.warning(f"Index Faiss {idx} hors limites (taille des chunks: {len(self.document_chunks)}).")

            results.sort(key=lambda x: x["score"], reverse=True)
            if len(results) > k:
                results = results[:k]

            if min_score is not None:
                logging.info(f"{len(results)} chunks pertinents trouvés (score minimum: {min_score * 100:.2f}%).")
            else:
                logging.info(f"{len(results)} chunks pertinents trouvés.")

            return results

        except MistralAPIException as e:
            logging.error(f"Erreur API Mistral lors de la génération de l'embedding de la requête: {e}")
            logging.error(f"  Détails: Status Code={e.status_code}, Message={e.message}")
            return []
        except Exception as e:
            logging.error(f"Erreur inattendue lors de la recherche: {e}")
            return []
