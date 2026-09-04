# Assistant RAG NBA — SportSee

Assistant virtuel basé sur Mistral AI, utilisant la technique de Retrieval-Augmented
Generation (RAG) pour répondre à des questions sur des archives textuelles NBA
(commentaires de matchs, discussions Reddit) à partir d'une base de connaissances
personnalisée.

## Fonctionnalités

- 🔍 **Recherche sémantique** avec FAISS (index `IndexFlatIP`, similarité cosinus)
- 📄 **Ingestion multi-format** : PDF (avec OCR automatique via EasyOCR pour les
  documents scannés), DOCX, TXT, CSV, XLSX
- 🤖 **Génération de réponses** avec les modèles Mistral (`mistral-small-latest`
  par défaut)
- 💬 **Interface conversationnelle** via Streamlit

## Prérequis

- Python 3.10+
- Clé API Mistral (obtenue sur [console.mistral.ai](https://console.mistral.ai/))

## Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/jackobess/OC_Projet10.git
cd OC_Projet10
```

2. **Créer et activer un environnement virtuel**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

> ⚠️ `easyocr` installe des dépendances lourdes (`torch`, `torchvision`).
> L'installation peut prendre plusieurs minutes selon la connexion.

4. **Configurer la clé API**

Créer un fichier `.env` à la racine du projet :

```
MISTRAL_API_KEY=votre_clé_api_mistral
```

C'est la seule variable sensible attendue dans `.env`. Le reste de la
configuration (modèles utilisés, taille des chunks, chemins des fichiers...)
est centralisé dans `utils/config.py`.

## Structure du projet

```
.
├── indexer.py              # Script d'indexation (chunking, embeddings, index FAISS)
├── MistralChat.py           # Application Streamlit (RAG + chat)
├── requirements.txt
├── .env                     # Clé API (non versionné)
├── inputs/                  # Documents sources (PDF, XLSX...)
├── vector_db/                # Index FAISS + chunks (généré, non versionné)
└── utils/
    ├── config.py            # Configuration centrale de l'application
    ├── data_loader.py       # Extraction de texte (PDF/OCR, DOCX, TXT, CSV, XLSX)
    └── vector_store.py       # Gestion de l'index FAISS et recherche sémantique
```

## Utilisation

### 1. Ajouter des documents

Placer les documents sources dans le dossier `inputs/`. Formats supportés :
PDF, DOCX, TXT, CSV, XLSX.

Les PDF scannés (peu ou pas de texte extractible) basculent automatiquement
sur un pipeline OCR (EasyOCR) — pas d'action manuelle requise.

### 2. Indexer les documents

```bash
python indexer.py
```

Ce script :
1. Charge et parse les documents depuis `inputs/`
2. Découpe les documents en chunks (`CHUNK_SIZE`/`CHUNK_OVERLAP`, voir `config.py`)
3. Génère les embeddings via l'API Mistral (`mistral-embed`)
4. Construit l'index FAISS et le sauvegarde dans `vector_db/`

> `vector_db/` n'est pas versionné dans Git : il est entièrement régénérable à
> partir de `inputs/` via cette commande. Relancer `python indexer.py` après
> tout ajout/modification de document source.

### 3. Lancer l'application

```bash
streamlit run MistralChat.py
```

Application accessible sur http://localhost:8501.

## Modules principaux

### `utils/data_loader.py`
Extraction de texte multi-format, avec fallback OCR (EasyOCR) automatique pour
les PDF scannés ou peu extractibles.

### `utils/vector_store.py`
Gère l'index vectoriel FAISS : découpage des documents en chunks, génération
des embeddings via l'API Mistral, création/chargement de l'index, recherche
par similarité cosinus.

### `utils/config.py`
Centralise la configuration de l'application (modèles Mistral, taille des
chunks, chemins des fichiers, paramètres de recherche).

## Personnalisation

Les paramètres suivants sont modifiables dans `utils/config.py` :
- Modèles Mistral utilisés (génération / embedding)
- Taille des chunks et chevauchement (`CHUNK_SIZE`, `CHUNK_OVERLAP`)
- Nombre de documents récupérés par recherche (`SEARCH_K`)
- Titre de l'application et nom du domaine (`APP_TITLE`, `NAME`)
