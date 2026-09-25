# MistralChat2.py          => streamlit run MistralChat2.py -- --season 2024-25
#
# Version LangChain : remplace le SDK natif mistralai (MistralClient/ChatMessage)
# par langchain_mistralai.ChatMistralAI pour la génération de réponse finale.
# Le reste (routage SQL, RAG, Logfire) est inchangé par rapport à MistralChat.py.

import streamlit as st
import streamlit.components.v1 as components
import os
import argparse
import configparser
import logging
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from utils.prompts import SQL_SYSTEM_PROMPT, SYSTEM_PROMPT, SYSTEM_PROMPT_2

# --- Importations depuis vos modules ---
try:
    from utils.config import (
        MISTRAL_API_KEY, MODEL_NAME, SEARCH_K,
        APP_TITLE, NAME, DATA_INFO_FILE ,VECTOR_DB_DIR
    )
    from utils.vector_store import VectorStoreManager
    from utils.sql_tool import nba_sql_tool, llm as sql_llm, set_season, has_season_data
except ImportError as e:
    st.error(f"Erreur d'importation: {e}. Vérifiez la structure de vos dossiers et les fichiers dans 'utils'.")
    st.stop()

import logfire
logfire.configure()
logfire.instrument_requests()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- Détermination de la saison de référence (--season, sinon data_info.ini) ---
def _parse_season_arg() -> str | None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", default=None)
    args, _ = parser.parse_known_args()
    return args.season


def _read_season_from_ini(path: str = DATA_INFO_FILE) -> str | None:
    config = configparser.ConfigParser()
    read_ok = config.read(path, encoding="utf-8")
    if read_ok and config.has_option("sources_info", "saison"):
        return config.get("sources_info", "saison")
    return None


SEASON = _parse_season_arg() or _read_season_from_ini()
if not SEASON:
    st.error("Aucune saison spécifiée (--season) et pas de saison trouvée dans data_info.ini. Arrêt.")
    st.stop()

if not has_season_data(SEASON):
    st.error(f"Aucune donnée en base (table 'stats') pour la saison '{SEASON}'. Arrêt.")
    logging.error(f"Saison '{SEASON}' absente de la base — arrêt du serveur.")
    st.stop()

set_season(SEASON)
logging.info(f"Saison de référence pour les stats: {SEASON}")

# --- Configuration du client LLM (LangChain / ChatMistralAI) ---
api_key = MISTRAL_API_KEY
model = MODEL_NAME

if not api_key:
    st.error("Erreur : Clé API Mistral non trouvée (MISTRAL_API_KEY). Veuillez la définir dans le fichier .env.")
    st.stop()

try:
    client = ChatMistralAI(model=model, mistral_api_key=api_key, temperature=0.1)
    logging.info("Client Mistral (LangChain) initialisé.")
except Exception as e:
    st.error(f"Erreur lors de l'initialisation du client Mistral : {e}")
    logging.exception("Erreur initialisation client Mistral")
    st.stop()

# --- Chargement du Vector Store (mis en cache) ---
@st.cache_resource # Garde le manager chargé en mémoire pour la session
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        # Vérifie si l'index a bien été chargé par le constructeur
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None # Retourne None si échec
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except FileNotFoundError:
         st.error("Fichiers d'index ou de chunks non trouvés.")
         st.warning("Veuillez exécuter 'python indexer.py' pour créer la base de connaissances.")
         logging.error("FileNotFoundError lors de l'init de VectorStoreManager.")
         return None
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None

vector_store_manager = get_vector_store_manager()

# --- Routage vers le Tool SQL (function calling LangChain/Mistral) ---
llm_with_tools = sql_llm.bind_tools([nba_sql_tool])

def route_and_maybe_run_sql(question: str) -> str | None:
    """Laisse le LLM décider si la question nécessite le Tool SQL.
    Retourne le résultat brut du tool si appelé, sinon None (=> RAG classique)."""
    with logfire.span("sql_routing", question=question) as span:
        ai_msg = llm_with_tools.invoke(question)
        tool_calls = ai_msg.tool_calls
        span.set_attribute("tool_used", bool(tool_calls))
        if not tool_calls:
            return None
        tool_call = tool_calls[0]
#        result = nba_sql_tool.invoke(tool_call["args"])
        result = nba_sql_tool.invoke(question)  # on passe la question originale, pas les args du tool, pour maitriser la logique de "saison"
        span.set_attribute("sql_result_preview", result[:300])
        return result

# --- Initialisation de l'historique de conversation ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les équipes, les joueurs ou les statistiques, et je vous répondrai en me basant sur les données les plus récentes."}]

# --- Fonctions ---

def generer_reponse(prompt_messages: list[HumanMessage]) -> str:
    if not prompt_messages:
        logging.warning("Tentative de génération de réponse avec un prompt vide.")
        return "Je ne peux pas traiter une demande vide."
    try:
        with logfire.span("mistral_generation", model=model, n_messages=len(prompt_messages)) as span:
            span.set_attribute("full_prompt", prompt_messages[0].content)
            logging.info(f"Appel à l'API Mistral modèle '{model}' avec {len(prompt_messages)} message(s).")
            response = client.invoke(prompt_messages)
            answer = response.content
            if answer:
                span.set_attribute("response_preview", answer[:200])
                if response.usage_metadata:
                    span.set_attribute("tokens_used", response.usage_metadata.get("total_tokens", 0))
                logging.info("Réponse reçue de l'API Mistral.")
                return answer
            else:
                logging.warning("L'API n'a pas retourné de contenu valide.")
                return "Désolé, je n'ai pas pu générer de réponse valide pour le moment."
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API Mistral: {e}")
        logging.exception("Erreur API Mistral pendant client.invoke")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard."


# --- Interface Utilisateur Streamlit ------------------------------------------------------------------------- Streamlit UI
#st.title(APP_TITLE)
#st.caption(f"Assistant virtuel pour {NAME} | Modèle: {model}")
#st.info(f"📅 Saison de référence pour les stats: {SEASON}")

# ================================================================================================================================ La side bar
with st.sidebar:
    st.title(f"🏀 {APP_TITLE} (v3)")
    st.info(f"📅 Saison: {SEASON}")
    st.divider()
    st.markdown("[📦 GitHub repo](https://github.com/jackobess/OC_Projet10)")
    st.markdown("[📖 README](https://github.com/jackobess/OC_Projet10/blob/main/README.md)")
    st.divider()
    st.caption(f"📁 Vector: {VECTOR_DB_DIR}")
    search_k = st.slider("SEARCH_K (nb chunks)", min_value=3, max_value=20, value=SEARCH_K, key="search_k")
    horsRAG = st.checkbox("Autoriser les réponses hors RAG", value=True)
    ThePrompt = SYSTEM_PROMPT if horsRAG else SYSTEM_PROMPT_2
# ================================================================================================================================ 

# Affichage des messages de l'historique (pour l'UI)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Zone de saisie utilisateur
if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    # 1. Ajouter et afficher le message de l'utilisateur
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # === Début de la logique de routage (SQL Tool en priorité, RAG en repli) ===

    # 2. Le LLM décide si la question est "chiffrée" (=> Tool SQL) ou pas
    sql_result = route_and_maybe_run_sql(prompt)

    if sql_result is not None and not sql_result.startswith("#SQLTOOL"):
        # Question chiffrée traitée avec succès par le Tool SQL : pas besoin du RAG
        # (on évite une recherche vectorielle inutile)
        logging.info("Question routée vers le Tool SQL.")
        final_prompt_for_llm = SQL_SYSTEM_PROMPT.format(question=prompt, sql_result=sql_result, season=SEASON)

    else:
        # Question qualitative (pas de tool call), OU Tool SQL déclenché mais en échec (#SQLTOOL) : RAG classique
        if sql_result is not None:
            logging.info(f"Tool SQL en échec, fallback RAG : {sql_result[:150]}")
        logging.info("Question routée vers le RAG.")

        # 2bis. Vérifier si le Vector Store est disponible
        if vector_store_manager is None:
            st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
            logging.error("VectorStoreManager non disponible pour la recherche.")
            # On arrête ici car on ne peut pas faire de RAG
            st.stop()

        # 3. Rechercher le contexte dans le Vector Store
        try:
            with logfire.span("RAG retrieval", question=prompt, k=search_k) as span:
                logging.info(f"Recherche de contexte pour la question: '{prompt}' avec k={search_k}")
                search_results = vector_store_manager.search(prompt, k=search_k)
                span.set_attribute("n_chunks_found", len(search_results))
                span.set_attribute("scores", [round(r['score'], 1) for r in search_results])
    #            span.set_attribute("sources", [r['metadata'].get('source', '?') for r in search_results])
                span.set_attribute("chunks_preview", [f"{r['metadata'].get('source', '?')}: {r['text'][:200]}" for r in search_results])
                logging.info(f"{len(search_results)} chunks trouvés dans le Vector Store.")
        except Exception as e:
            st.error(f"Une erreur est survenue lors de la recherche d'informations pertinentes: {e}")
            logging.exception(f"Erreur pendant vector_store_manager.search pour la query: {prompt}")
            search_results = [] # On continue sans contexte si la recherche échoue

        # 4. Formater le contexte pour le prompt LLM
        context_str = "\n\n----------------------------\n".join([
            f"Source: {res['metadata'].get('source', 'Inconnue')} (Score: {res['score']:.1f}%)\nContenu: {res['text']}"
            for res in search_results
        ])

        if not search_results:
            context_str = "Aucune information pertinente trouvée dans la base de connaissances pour cette question."
            logging.warning(f"Aucun contexte trouvé pour la query: {prompt}")

        # 5. Construire le prompt final pour l'API Mistral en utilisant le System Prompt RAG
        sql_attempt_note = f"[NOTE INTERNE - à ignorer dans la réponse: tentative SQL infructueuse >> {sql_result}]\n\n" if sql_result else ""
        final_prompt_for_llm = ThePrompt.format(context_str=sql_attempt_note + context_str, question=prompt, season=SEASON)

    # Créer la liste de messages pour l'API (juste le prompt système/utilisateur combiné)
    messages_for_api = [
        HumanMessage(content=final_prompt_for_llm)
    ]

    # === Fin de la logique RAG ===

    # 6. Afficher indicateur + Générer la réponse de l'assistant via LLM
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...") # Indicateur simple

        # Génération de la réponse de l'assistant en utilisant le prompt augmenté
        response_content = generer_reponse(messages_for_api)

        # Affichage de la réponse complète
        message_placeholder.write(response_content)

        # Menu déroulant pour inspecter le prompt envoyé au LLM    ************************* prompt view
        with st.expander("🔍 Voir le prompt / contexte envoyé au LLM"):
        #    st.code(final_prompt_for_llm, language="text")
        #    st.caption(final_prompt_for_llm)
            formatted_prompt = final_prompt_for_llm.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            st.markdown(
                f"""
                <div style="
                    background-color: #2b2b2b;
                    color: #dcdcdc;
                    font-family: 'Courier New', Courier, monospace;
                    font-size: 9px;
                    letter-spacing: -0.3px;
                    line-height: 1.2;
                    padding: 10px;
                    border-radius: 6px;
                    overflow-x: auto;
                    white-space: pre-wrap;
                ">
                    {formatted_prompt}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Force l'Auto-scroll vers la réponse
        st.markdown('<div id="end-of-chat"></div>', unsafe_allow_html=True)
        components.html(
            """
            <script>
                var element = window.parent.document.getElementById("end-of-chat");
                if (element) { element.scrollIntoView({behavior: "smooth"}); }
            </script>
            """,
            height=0
        )

    # 7. Ajouter la réponse de l'assistant à l'historique (pour affichage UI)
    st.session_state.messages.append({"role": "assistant", "content": response_content})

# Petit pied de page optionnel
st.markdown("---")
st.caption("Powered by Mistral AI (via LangChain) & Faiss | Data-driven NBA Insights")
