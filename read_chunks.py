# read_chunks.py
"""
Petit utilitaire pour lire document_chunks.pkl de façon lisible.
Usage : python read_chunks.py [numero_du_chunk]
    - Sans argument : affiche un résumé (nombre de chunks, sources, tailles)
    - Avec un numéro : affiche le contenu complet de ce chunk
"""
import pickle
import sys
from collections import Counter
from utils.config import DOCUMENT_CHUNKS_FILE

with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

def meta(c):
    return c.metadata.model_dump()

if len(sys.argv) > 1:
    idx = int(sys.argv[1])
    chunk = chunks[idx]
    print(f"--- Chunk #{idx} ---")
    print(f"ID: {chunk.id}")
    print(f"Source: {meta(chunk).get('source', 'N/A')}")
    print(f"Metadata: {meta(chunk)}")
    print(f"Chunk dans doc: {chunk.chunk_id_in_doc} (start_index={chunk.start_index})")
    print(f"\n--- Texte ({len(chunk.text)} caractères) ---")
    print(chunk.text)
else:
    print(f"Nombre total de chunks : {len(chunks)}\n")
    sources = Counter(meta(c).get("source", "N/A") for c in chunks)
    print("Répartition par source :")
    for source, count in sources.most_common():
        print(f"  {count:3d} chunks - {source}")
    print("\nPour voir un chunk : python read_chunks.py <numero>")