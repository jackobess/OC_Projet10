# read_chunks.py
"""
Petit utilitaire pour lire un chunk extrait de document_chunks.pkl
Usage : python read_chunks.py [-s] [numero_du_chunk | chaine à chercher]
    - Sans argument : résumé (nombre de chunks, répartition par source)
    - Avec un numéro : affiche le contenu complet de ce chunk
    - Avec une chaîne : liste les chunks contenant cette sous-chaîne
      (insensible à la casse) avec un extrait autour de la 1re occurrence
    - Avec -s : force la recherche, même si l'argument est un nombre
"""
import argparse
import pickle
from collections import Counter
from utils.config import DOCUMENT_CHUNKS_FILE

with open(DOCUMENT_CHUNKS_FILE, "rb") as f:
    chunks = pickle.load(f)

def meta(c):
    return c.metadata.model_dump()

def show_chunk(idx):
    chunk = chunks[idx]
    print(f"--- Chunk #{idx} ---")
    print(f"ID: {chunk.id}")
    print(f"Source: {meta(chunk).get('source', 'N/A')}")
    print(f"Metadata: {meta(chunk)}")
    print(f"Chunk dans doc: {chunk.chunk_id_in_doc} (start_index={chunk.start_index})")
    print(f"\n--- Texte ({len(chunk.text)} caractères) ---")
    print(chunk.text)

def search_chunks(needle, context=60):
    needle_l = needle.lower()
    hits = []
    for i, c in enumerate(chunks):
        text_l = c.text.lower()
        pos = text_l.find(needle_l)
        if pos != -1:
            start = max(0, pos - context)
            end = min(len(c.text), pos + len(needle) + context)
            snippet = c.text[start:end].replace("\n", " ")
            hits.append((i, c, text_l.count(needle_l), snippet))

    print(f"{len(hits)} chunk(s) contenant « {needle} »\n")
    for i, c, n, snippet in hits:
        print(f"#{i:4d} | {meta(c).get('source', 'N/A')} | {n} occ. | ...{snippet}...")
    if hits:
        print("\nPour voir un chunk : python read_chunks.py <numero>")

def summary():
    print(f"Nombre total de chunks : {len(chunks)}\n")
    sources = Counter(meta(c).get("source", "N/A") for c in chunks)
    print("Répartition par source :")
    for source, count in sources.most_common():
        print(f"  {count:3d} chunks - {source}")
    print("\nPour voir un chunk : python read_chunks.py <numero>")
    print('Pour chercher      : python read_chunks.py [-s] "texte à chercher"')

parser = argparse.ArgumentParser(description="Lire ou chercher un chunk.")
parser.add_argument("query", nargs="*", help="numéro de chunk ou chaîne à chercher")
parser.add_argument("-s", "--search", action="store_true",
                    help="force la recherche de sous-chaîne (même pour un nombre)")
args = parser.parse_args()

arg = " ".join(args.query)
if not arg:
    summary()
elif arg.isdigit() and not args.search:
    idx = int(arg)
    if idx >= len(chunks):
        parser.exit(1, f"Index {idx} hors limites (0 à {len(chunks) - 1})\n")
    show_chunk(idx)
else:
    search_chunks(arg)
    