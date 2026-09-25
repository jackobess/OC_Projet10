# evaluate_ragas.py         => python evaluate_ragas.py [--season 2024-25] [--output ragas_results.csv]

"""
Évalue le prototype RAG/SQL actuel (routing SQL-first + fallback RAG Faiss + génération Mistral)
avec RAGAS sur ragas_dataset.py.

Usage :
    python evaluate_ragas.py
    python evaluate_ragas.py --season 2024-25 --output ragas_data/ragas_results_v2.xlsx

Prérequis (voir requirements.txt) :
    ragas==0.2.15, datasets==3.2.0, langchain-mistralai==0.2.12, orjson<3.11
    (langchain-mistralai est nécessaire pour que RAGAS utilise Mistral comme juge au lieu d'OpenAI par défaut)

Ce script réplique EXACTEMENT la logique de MistralChat.py :
  1. Le LLM décide (function calling) si la question doit passer par le Tool SQL (nba_sql_tool)
  2. Si oui et succès -> SQL_SYSTEM_PROMPT (résultat SQL comme "contexte" pour RAGAS)
  3. Sinon (pas de tool call, ou #SQLTOOL en échec) -> RAG classique (recherche Faiss + SYSTEM_PROMPT)

La route empruntée par question ("sql" / "rag") est ajoutée au CSV de sortie pour pouvoir
splitter les scores RAGAS par route en plus de par catégorie dans le rapport.

NB : les métriques context_precision / context_recall sont conçues pour du texte récupéré par
similarité vectorielle. Sur la route "sql", le "contexte" est le résultat brut de la requête SQL
(une ligne de données, pas un passage de texte) — ces deux métriques y sont donc moins
interprétables. Garder faithfulness / answer_correctness comme lecture principale pour la
catégorie/route SQL dans le rapport.
"""

import argparse
import configparser
import logging

import pandas as pd
from datasets import Dataset
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

from utils.config import MISTRAL_API_KEY, MODEL_NAME, SEARCH_K, DATA_INFO_FILE
from utils.vector_store import VectorStoreManager
from utils.sql_tool import nba_sql_tool, llm as sql_llm, set_season, has_season_data
from ragas_data.ragas_dataset import RAGAS_DATASET
from utils.prompts import SQL_SYSTEM_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_2
import time
import random

def with_retry(fn, *args, max_retries=6, base_delay=5, **kwargs):
    """Retry avec backoff exponentiel sur les 429 / erreurs réseau transitoires."""
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            retryable = "429" in msg or "Too Many Requests" in msg or "rate limit" in msg.lower() \
                        or "503" in msg or "timeout" in msg.lower()
            if not retryable or attempt == max_retries:
                raise
            delay = min(base_delay * 2 ** attempt, 60) + random.uniform(0, 2)
            logging.warning(f"    429/erreur transitoire (tentative {attempt + 1}/{max_retries}) -> pause {delay:.0f}s")
            time.sleep(delay)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Résolution de la saison (--season, sinon data_info.ini), même logique que MistralChat.py ---
def _parse_season_arg() -> tuple[str | None, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default=None)
    parser.add_argument("--output", type=str, default="ragas_data/ragas_results.xlsx")
    args, _ = parser.parse_known_args()
    return args.season, args.output


def _read_season_from_ini(path: str = DATA_INFO_FILE) -> str | None:
    config = configparser.ConfigParser()
    read_ok = config.read(path, encoding="utf-8")
    if read_ok and config.has_option("sources_info", "saison"):
        return config.get("sources_info", "saison")
    return None

def route_and_maybe_run_sql(llm_with_tools, question: str) -> str | None:
    """Même logique que MistralChat.py : laisse le LLM décider via function calling."""
    ai_msg = llm_with_tools.invoke(question)
    tool_calls = ai_msg.tool_calls
    if not tool_calls:
        return None
    tool_call = tool_calls[0]
#    return nba_sql_tool.invoke(tool_call["args"])
    return nba_sql_tool.invoke(question)  # on passe la question originale, pas les args du tool, pour maitriser la logique de season dans le prompt SQL

def run_agent_pipeline(vector_store, client, llm_with_tools, question: str, season: str):
    """Reproduit exactement le flow route -> (SQL | RAG) -> prompt -> génération de MistralChat.py.
    Retourne (answer, contexts, route) où route in {"sql", "rag"}."""
    sql_result = route_and_maybe_run_sql(llm_with_tools, question)

    if sql_result is not None and not sql_result.startswith("#SQLTOOL"):
        route = "sql"
        contexts = [sql_result]
        final_prompt = SQL_SYSTEM_PROMPT.format(question=question, sql_result=sql_result, season=season)
    else:
        route = "rag_fallback_after_sql_fail" if sql_result is not None else "rag_no_tool_call"
        sql_attempt_note = (
            f"[NOTE INTERNE - à ignorer dans la réponse: tentative SQL infructueuse >> {sql_result}]\n\n"
            if sql_result else ""
        )
        logging.info(f"*********QUERY EXACTE: {repr(question)}")
        search_results = vector_store.search(question, k=SEARCH_K)
        if search_results:
            contexts = [res["text"] for res in search_results]
            context_str = "\n\n----------------------------\n".join(
                f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
                for res in search_results
            )
        else:
            contexts = ["(aucun contexte récupéré)"]
            context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
        final_prompt = SYSTEM_PROMPT.format(context_str=sql_attempt_note + context_str, question=question)

    response = client.chat(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content=final_prompt)],
        temperature=0.1,
    )
    answer = response.choices[0].message.content if response.choices else ""
    return answer, contexts, route


def build_ragas_dataset(golden_dataset, vector_store, client, llm_with_tools, season):
    """Fait tourner chaque question du ragas dataset dans le pipeline agent, collecte answer + contexts + route."""
    rows = []
    for i, item in enumerate(golden_dataset):
        question = item["question"]
        if question.startswith("TODO"):
            logging.warning(f"[{i}] Question non renseignée (TODO), ignorée : {question[:60]}...")
            continue

        logging.info(f"[{i}] ({item['category']}) {question}")
        answer, contexts, route = with_retry(run_agent_pipeline, vector_store, client, llm_with_tools, question, season)
        time.sleep(1)
        logging.info(f"    -> route: {route}")

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
            "category": item["category"],
            "route": route,
        })
    return rows


def main():
    season_arg, output = _parse_season_arg()
    season = season_arg or _read_season_from_ini()
    if not season:
        raise SystemExit("Aucune saison spécifiée (--season) et pas de saison trouvée dans data_info.ini. Arrêt.")

    if not MISTRAL_API_KEY:
        raise SystemExit("MISTRAL_API_KEY manquante dans .env, impossible de lancer l'évaluation.")

    if not has_season_data(season):
        raise SystemExit(f"Aucune donnée en base (table 'stats') pour la saison '{season}'. Arrêt.")

    set_season(season)
    logging.info(f"Saison de référence pour les stats: {season}")

    logging.info("Chargement du vector store...")
    vector_store = VectorStoreManager()
    if vector_store.index is None:
        raise SystemExit("Index Faiss non chargé. Lance d'abord 'python indexer.py'.")

    client = MistralClient(api_key=MISTRAL_API_KEY)
    llm_with_tools = sql_llm.bind_tools([nba_sql_tool])

    logging.info(f"Exécution du pipeline agent (SQL + RAG) sur {len(RAGAS_DATASET)} questions du ragas dataset...")
    rows = build_ragas_dataset(RAGAS_DATASET, vector_store, client, llm_with_tools, season)

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
        run_config=RunConfig(max_workers=2)
    )

    scores_df = result.to_pandas()
    scores_df["category"] = [r["category"] for r in rows]
    scores_df["route"] = [r["route"] for r in rows]

    scores_df.to_excel(output, index=False)
    logging.info(f"Résultats détaillés sauvegardés dans {output}")

    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "answer_correctness"]

    print("\n=== Scores moyens globaux ===")
    print(scores_df[metric_cols].mean().round(3))

    print("\n=== Scores moyens par catégorie ===")
    print(scores_df.groupby("category")[metric_cols].mean().round(3))

    print("\n=== Scores moyens par route (sql vs rag) ===")
    print(scores_df.groupby("route")[metric_cols].mean().round(3))


if __name__ == "__main__":
    main()