# evaluate_ragas.py
"""
Évalue le prototype RAG actuel (search Faiss + génération Mistral) avec RAGAS sur le ragas dataset ragas_dataset_1.py.

Usage :
    python evaluate_ragas_1.py
    python evaluate_ragas_1.py --output ragas_results_v1.csv

Prérequis (voir requirements.txt) :
    ragas==0.2.15, datasets==3.2.0, langchain-mistralai==0.2.12, orjson<3.11
    (langchain-mistralai est nécessaire pour que RAGAS utilise Mistral comme juge au lieu d'OpenAI par défaut)

Ce script réplique la logique RAG de MistralChat.py (search + prompt + génération)
sans dépendre de Streamlit, pour pouvoir tourner en batch sur tout le ragas dataset.
"""
import argparse
import logging

import pandas as pd
from datasets import Dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K
from utils.vector_store import VectorStoreManager
from ragas_dataset_1 import RAGAS_DATASET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Même prompt système que MistralChat.py, pour évaluer le pipeline tel qu'il tourne réellement
SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.
Ta mission est de répondre aux questions des fans en animant le débat.

---
{context_str}
---

QUESTION DU FAN:
{question}

RÉPONSE DE L'ANALYSTE NBA:"""


def run_rag_pipeline(vector_store: VectorStoreManager, client: MistralClient, question: str):
    """Reproduit exactement le flow search -> prompt -> génération de MistralChat.py."""
    search_results = vector_store.search(question, k=SEARCH_K)
    contexts = [res["text"] for res in search_results]

    if search_results:
        context_str = "\n\n---\n\n".join(
            f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in search_results
        )
    else:
        context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."

    prompt = SYSTEM_PROMPT.format(context_str=context_str, question=question)

    response = client.chat(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.1,
    )
    answer = response.choices[0].message.content if response.choices else ""
    return answer, contexts


def build_ragas_dataset(golden_dataset, vector_store, client):
    """Fait tourner chaque question du ragas dataset dans le RAG, collecte answer + contexts."""
    rows = []
    for i, item in enumerate(golden_dataset):
        question = item["question"]
        if question.startswith("TODO"):
            logging.warning(f"[{i}] Question non renseignée (TODO), ignorée : {question[:60]}...")
            continue

        logging.info(f"[{i}] ({item['category']}) {question}")
        answer, contexts = run_rag_pipeline(vector_store, client, question)

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts if contexts else ["(aucun contexte récupéré)"],
            "ground_truth": item["ground_truth"],
            "category": item["category"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Évaluation RAGAS du prototype RAG")
    parser.add_argument("--output", type=str, default="ragas_results.csv",
                        help="Fichier CSV de sortie pour les résultats détaillés")
    args = parser.parse_args()

    if not MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY manquante dans .env, impossible de lancer l'évaluation.")

    logging.info("Chargement du vector store...")
    vector_store = VectorStoreManager()
    if vector_store.index is None:
        raise SystemExit("Index Faiss non chargé. Lance d'abord 'python indexer.py'.")

    client = MistralClient(api_key=MISTRAL_API_KEY)

    logging.info(f"Exécution du pipeline RAG sur {len(RAGAS_DATASET)} questions du ragas dataset...")
    rows = build_ragas_dataset(RAGAS_DATASET, vector_store, client)

    # Dataset RAGAS attend ces colonnes exactement : question, answer, contexts, ground_truth
    ragas_dataset = Dataset.from_list([
        {
            "question": r["question"],
            "answer": r["answer"],
            "contexts": r["contexts"],
            "ground_truth": r["ground_truth"],
        }
        for r in rows
    ])

    # RAGAS utilise OpenAI par défaut comme juge pour ses métriques ; on le bascule sur
    # Mistral (LLM + embeddings) pour rester cohérent avec le reste du projet et éviter
    # de dépendre d'une clé OpenAI qu'on n'a pas.
    ragas_llm = LangchainLLMWrapper(ChatMistralAI(model=MODEL_NAME, mistral_api_key=MISTRAL_API_KEY))
    ragas_embeddings = LangchainEmbeddingsWrapper(MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY))

    logging.info("Lancement de l'évaluation RAGAS (peut prendre plusieurs minutes)...")
    result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    scores_df = result.to_pandas()
    scores_df["category"] = [r["category"] for r in rows]

    scores_df.to_csv(args.output, index=False)
    logging.info(f"Résultats détaillés sauvegardés dans {args.output}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]
    print("\n=== Scores moyens globaux ===")
    print(scores_df[metric_cols].mean().round(3))

    print("\n=== Scores moyens par catégorie ===")
    print(scores_df.groupby("category")[metric_cols].mean().round(3))


if __name__ == "__main__":
    main()
