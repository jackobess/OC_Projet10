# utils/schemas.py
"""
Modèles Pydantic qui valident chaque étape du pipeline de préparation des données :

    extraction brute        -> RawDocument
    nettoyage (Pydantic AI) -> CleanedText
    chunking                -> TextChunk
    embedding                -> EmbeddedChunk
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentMetadata(BaseModel):
    """Métadonnées communes à un document / chunk. `extra='allow'` pour rester compatible
    avec les clés déjà utilisées ailleurs dans le projet (sheet, category, etc.)."""

    model_config = ConfigDict(extra="allow")

    source: str
    filename: str
    category: str
    full_path: str
    sheet: Optional[str] = None


class RawDocument(BaseModel):
    """Document tel que sorti de data_loader.py, avant tout traitement."""

    page_content: str = Field(..., min_length=1)
    metadata: DocumentMetadata

    @field_validator("page_content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("page_content vide après strip()")
        return v


class CleanedText(BaseModel):
    """Sortie de l'étape de nettoyage (Pydantic AI). Un objet par RawDocument."""

    cleaned_content: str = Field(..., min_length=1)
    removed_noise: bool = False
    notes: Optional[str] = None
    metadata: DocumentMetadata

    @field_validator("cleaned_content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("cleaned_content vide après nettoyage")
        return v


class TextChunk(BaseModel):
    """Chunk validé avant embedding."""

    id: str
    text: str = Field(..., min_length=1)
    metadata: DocumentMetadata
    chunk_id_in_doc: int
    start_index: int = -1

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chunk text vide")
        return v


class EmbeddedChunk(BaseModel):
    """Chunk + vecteur, validé juste avant écriture dans Faiss."""

    id: str
    text: str
    metadata: DocumentMetadata
    embedding: List[float]

    @field_validator("embedding")
    @classmethod
    def check_embedding(cls, v: List[float]) -> List[float]:
        if not v:
            raise ValueError("embedding vide")
        if any(x != x for x in v):  # NaN check (x != x est True uniquement pour NaN)
            raise ValueError("embedding contient des NaN")
        return v
