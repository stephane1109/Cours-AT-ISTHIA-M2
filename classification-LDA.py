# pip install streamlit pandas spacy gensim nltk pyLDAvis matplotlib
# python -m spacy download fr_core_news_md

# Lancer : python -m streamlit run main.py

# -*- coding: utf-8 -*-
# Streamlit (mode wide) : Classification NON supervisée des MOTS par LDA
# Spécifications :
#   - Classement des MOTS (pas des documents) : attribution du topic dominant à chaque mot.
#   - Barre latérale : no_below (docs min), no_above (%) et K (manuel uniquement).
#   - Stopwords français appliqués.
#   - Top-mots par topic avec explication du "poids".
#   - Attribution mots->topic avec explication du "score_topic".
#   - Visualisation PyLDAvis + export HTML + légende couleurs + explication du curseur λ.
#   - Export CSV : top-mots par topic, attribution mots→topic, probabilités P(mot|topic).

import re
import io
import time
import contextlib
from typing import List, Tuple

import streamlit as st
import pandas as pd
import numpy as np

# NLTK : stopwords FR
import nltk
from nltk.corpus import stopwords as nltk_stopwords

# scikit-learn : LDA et sac de mots
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# PyLDAvis : visualisation (sans sous-module sklearn)
import pyLDAvis

# ============================
# Configuration Streamlit
# ============================

st.set_page_config(page_title="Classification de mots par LDA (FR)", layout="wide")

# ============================
# Initialisation NLTK (FR)
# ============================

def initialiser_nltk_fr():
    """S’assurer que les ressources NLTK nécessaires pour le français sont disponibles."""
    with contextlib.suppress(LookupError):
        nltk.data.find("corpora/stopwords")
    try:
        _ = nltk_stopwords.words("french")
    except LookupError:
        nltk.download("stopwords")

initialiser_nltk_fr()

# ============================
# Utilitaires E/S
# ============================

def decoder_bytes_texte(b: bytes) -> str:
    """Décoder des bytes en texte (UTF-8 prioritaire, sinon tolérant)."""
    try:
        return b.decode("utf-8-sig")
    except UnicodeDecodeError:
        return b.decode("utf-8", errors="ignore")

def charger_uploads(fichiers) -> List[Tuple[str, str]]:
    """Charger des fichiers uploadés -> liste (nom_fichier, texte nettoyé)."""
    docs = []
    for f in fichiers or []:
        try:
            t = decoder_bytes_texte(f.read())
            t = re.sub(r"<[^>]+>", " ", t)       # suppression de balises éventuelles
            t = re.sub(r"\s+", " ", t).strip()   # normalisation espaces
            if t:
                docs.append((f.name, t))
        except Exception:
            continue
    return docs

def dedupliquer_documents(docs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Dédoublonner par nom de fichier (on garde la dernière version)."""
    d = {}
    for nom, texte in docs:
        d[nom] = texte
    return [(n, d[n]) for n in sorted(d.keys())]

# ============================
# LDA avec K manuel
# ============================

def construire_vecteurs_lda(textes: List[str],
                            stopwords_fr: set,
                            no_below_docs: int,
                            no_above_pct: float,
                            k_manuel: int,
                            random_state: int = 42):
    """
    Vectoriser les textes en appliquant les stopwords FR et les filtres no_below / no_above.
    no_below_docs -> min_df (entier), no_above_pct -> max_df (proportion).
    Ajuster l’LDA avec K défini manuellement.
    Renvoie (vect, X_counts, lda, k_utilise).
    """
    max_df_prop = max(0.5, min(1.0, no_above_pct / 100.0))  # borne de sécurité
    vect = CountVectorizer(
        lowercase=True,
        stop_words=list(stopwords_fr) if len(stopwords_fr) > 0 else None,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[^\W\d_]{2,}\b",  # uniquement lettres, longueur >= 2
        min_df=max(1, int(no_below_docs)),
        max_df=max_df_prop
    )
    try:
        X = vect.fit_transform(textes)
    except ValueError:
        X = None

    if X is None or X.shape[1] == 0:
        st.warning("Vos réglages no_below / no_above ont vidé le vocabulaire. Repli sur min_df=1 et max_df=1.0.")
        vect = CountVectorizer(
            lowercase=True,
            stop_words=list(stopwords_fr) if len(stopwords_fr) > 0 else None,
            strip_accents="unicode",
            token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
            min_df=1,
            max_df=1.0
        )
        X = vect.fit_transform(textes)
        if X.shape[1] == 0:
            raise ValueError("Vectorisation vide. Le corpus est trop limité ou non textuel.")

    k_utilise = max(2, int(k_manuel))
    lda = LatentDirichletAllocation(
        n_components=k_utilise,
        random_state=random_state,
        learning_method="batch",
        max_iter=25
    )
    lda.fit(X)
    return vect, X, lda, k_utilise

# ============================
# Extraction des résultats
# ============================

def extraire_top_mots_par_topic_sans_rang(lda: LatentDirichletAllocation,
                                          vect: CountVectorizer,
                                          nb_mots: int = 10) -> pd.DataFrame:
    """Tableau des top-mots par topic (colonnes : topic, mot, poids) — sans la colonne 'rang'."""
    termes = np.array(vect.get_feature_names_out())
    lignes = []
    for k, comp in enumerate(lda.components_):
        idx = np.argsort(comp)[::-1][:nb_mots]
        for j in idx:
            lignes.append({"topic": k, "mot": termes[j], "poids": float(comp[j])})
    df = pd.DataFrame(lignes, columns=["topic", "mot", "poids"])
    if len(df) > 0:
        df = df.sort_values(["topic", "poids"], ascending=[True, False]).reset_index(drop=True)
    return df

def classer_mots_par_topic(lda: LatentDirichletAllocation, vect: CountVectorizer) -> pd.DataFrame:
    """
    Attribution du topic dominant à chaque MOT du vocabulaire.
    On utilise lda.components_ (scores topic->mot). Score normalisé par colonne pour lisibilité.
    """
    termes = np.array(vect.get_feature_names_out())
    comp = np.asarray(lda.components_, dtype=float)   # (K, V)
    col_sums = comp.sum(axis=0) + 1e-12
    comp_norm = comp / col_sums
    topics_attribues = np.argmax(comp, axis=0).astype(int)
    scores = comp_norm[topics_attribues, np.arange(comp.shape[1])]
    df = pd.DataFrame({
        "mot": termes,
        "topic_attribue": topics_attribues,
        "score_topic": scores.astype(float)
    })
    df = df.sort_values(["topic_attribue", "score_topic"], ascending=[True, False]).reset_index(drop=True)
    return df

def calculer_probabilites_topic_mot(lda: LatentDirichletAllocation, vect: CountVectorizer) -> pd.DataFrame:
    """
    Construire la matrice des probabilités P(mot|topic).
    On normalise chaque ligne de lda.components_ pour obtenir une distribution de probabilité sur le vocabulaire.
    Renvoie un DataFrame long : colonnes = topic, mot, p_w_given_t.
    """
    termes = np.array(vect.get_feature_names_out())
    topic_term = np.asarray(lda.components_, dtype=float)               # (K, V)
    topic_term_probs = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)
    lignes = []
    for k in range(topic_term_probs.shape[0]):
        p = topic_term_probs[k, :]
        lignes.append(pd.DataFrame({"topic": k, "mot": termes, "p_w_given_t": p.astype(float)}))
    df = pd.concat(lignes, ignore_index=True)
    df = df.sort_values(["topic", "p_w_given_t"], ascending=[True, False]).reset_index(drop=True)
    return df

# ============================
# PyLDAvis (sans pyLDAvis.sklearn)
# ============================

def preparer_pyldavis_depuis_sklearn(lda: LatentDirichletAllocation, X_counts, vect: CountVectorizer):
    """Préparer les données PyLDAvis (projection mmds)."""
    topic_term = np.asarray(lda.components_, dtype=float)
    topic_term_dists = topic_term / (topic_term.sum(axis=1, keepdims=True) + 1e-12)

    doc_topic = np.asarray(lda.transform(X_counts), dtype=float)
    doc_topic_dists = doc_topic / (doc_topic.sum(axis=1, keepdims=True) + 1e-12)

    doc_lengths = np.asarray(X_counts.sum(axis=1)).ravel().astype(int)
    term_frequency = np.asarray(X_counts.sum(axis=0)).ravel().astype(int)
    vocab = np.array(vect.get_feature_names_out())

    mask_valid = np.logical_and(doc_lengths > 0, np.isfinite(doc_topic_dists).all(axis=1))
    doc_topic_dists = doc_topic_dists[mask_valid]
    doc_lengths = doc_lengths[mask_valid]

    vis = pyLDAvis.prepare(
        topic_term_dists=topic_term_dists,
        doc_topic_dists=doc_topic_dists,
        doc_lengths=doc_lengths.tolist(),
        vocab=vocab.tolist(),
        term_frequency=term_frequency.tolist(),
        mds="mmds"
    )
    return vis

def telecharger_df(df: pd.DataFrame, nom_fichier: str):
    """Téléchargement CSV pour une DataFrame."""
    st.download_button(
        label=f"Télécharger {nom_fichier}",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=nom_fichier,
        mime="text/csv",
        use_container_width=True
    )

# ============================
# Interface Streamlit
# ============================

st.title("Classification non supervisée des mots par LDA (FR)")

st.markdown(
    """
LDA (Latent Dirichlet Allocation) est un modèle probabiliste non-supervisé qui décompose un corpus en topics (themes).\n
Chaque topic est décrit par une distribution de probabilité de mots P(mot | topic) et, lorsqu’on projette un document, par une distribution de probabilité de topics P(topic | document).\n
Ici, on classe les mots du vocabulaire en leur attribuant le topic pour lequel P(mot | topic) est la plus élevée.\n
LDA donne des résultats plus stables lorsque le corpus est volumineux et hétérogène.\n
Le symbole « | » se lit « sachant ». P(mot | topic) signifie « probabilité d’observer ce mot sachant que l’on est dans ce topic ».
"""
)

if "docs_cumules" not in st.session_state:
    st.session_state["docs_cumules"] = []  # liste (nom, texte)

with st.sidebar:
    st.header("Paramètres du vocabulaire")
    no_below_docs = st.number_input(
        "no_below (docs min contenant le terme)",
        min_value=1, max_value=100, value=2, step=1,
        help="Conserve les termes apparaissant dans au moins ce nombre de documents (analogue à min_df)."
    )
    no_above_pct = st.slider(
        "no_above (%)",
        min_value=50, max_value=100, value=95, step=1,
        help="Supprime les termes apparaissant dans plus de ce pourcentage de documents (analogue à max_df)."
    )

    st.header("Nombre de topics (K)")
    k_manuel = st.number_input("K (topics)", min_value=2, max_value=50, value=10, step=1)

st.markdown("---")

fichiers = st.file_uploader("Déposer des fichiers .txt", type=["txt"], accept_multiple_files=True)
col_a, col_b, col_c = st.columns(3)
with col_a:
    if st.button("Ajouter à la liste", use_container_width=True):
        nouveaux = charger_uploads(fichiers)
        st.session_state["docs_cumules"].extend(nouveaux)
        st.session_state["docs_cumules"] = dedupliquer_documents(st.session_state["docs_cumules"])
with col_b:
    if st.button("Vider la liste", use_container_width=True):
        st.session_state["docs_cumules"] = []
with col_c:
    lancer = st.button("Lancer l’analyse", type="primary", use_container_width=True)

st.caption(f"Fichiers cumulés : {len(st.session_state['docs_cumules'])}")

if len(st.session_state["docs_cumules"]) > 0:
    # Aperçu du corpus : suppression de la colonne "nb_caracteres" comme demandé
    df_docs = pd.DataFrame([{"document": n, "nb_mots_estime": len(t.split())}
                            for n, t in st.session_state["docs_cumules"]])
    st.dataframe(df_docs, use_container_width=True, height=220)

if 'lancer' in locals() and lancer:
    debut = time.time()

    documents = st.session_state["docs_cumules"]
    if len(documents) == 0:
        st.error("Aucun document. Ajoutez des fichiers, puis relancez.")
        st.stop()

    textes = [t for _, t in documents]

    try:
        sw_fr = set(nltk_stopwords.words("french"))
    except LookupError:
        sw_fr = set()

    try:
        vect, X_counts, lda, k_utilise = construire_vecteurs_lda(
            textes=textes,
            stopwords_fr=sw_fr,
            no_below_docs=int(no_below_docs),
            no_above_pct=float(no_above_pct),
            k_manuel=int(k_manuel),
            random_state=42
        )
    except Exception as e:
        st.error(f"Impossible d’appliquer LDA : {e}")
        st.stop()

    st.success(f"Modèle LDA ajusté avec K = {k_utilise}")

    st.subheader("Top-mots par topic")
    st.markdown(
        "La colonne « poids » correspond à la valeur de la composante topic→mot dans la matrice interne du modèle (lda.components_). "
        "Plus le poids est élevé, plus le mot a tendance à être généré par ce topic. "
        "Le poids n’est pas une probabilité et peut dépasser 1 ; il dépend de la taille et de la composition du corpus. "
        "Pour des valeurs bornées entre 0 et 1, reportez-vous aux probabilités P(mot | topic) plus bas."
    )
    df_topics = extraire_top_mots_par_topic_sans_rang(lda, vect, nb_mots=10)
    st.dataframe(df_topics, use_container_width=True, height=360)
    telecharger_df(df_topics, "lda_top_mots_par_topic.csv")

    st.subheader("Attribution des mots au topic dominant")
    st.markdown(
        "La colonne « score_topic » indique, pour chaque mot, la part normalisée attribuée au topic dominant. "
        "Elle est calculée comme la valeur de la composante du topic choisi divisée par la somme des composantes de ce mot sur tous les topics. "
        "Le score est donc une proportion comprise entre 0 et 1 ; plus elle est proche de 1, plus le mot est spécifiquement associé à ce topic."
    )
    df_mots = classer_mots_par_topic(lda, vect)
    st.dataframe(df_mots, use_container_width=True, height=420)
    telecharger_df(df_mots, "lda_attribution_mots.csv")

    st.subheader("Probabilités P(mot | topic)")
    st.markdown(
        "Ce tableau fournit les probabilités apprises par le modèle. "
        "Pour chaque topic, la somme des probabilités sur tous les mots vaut 1."
    )
    df_probs = calculer_probabilites_topic_mot(lda, vect)
    ap = df_probs.groupby("topic").head(10).reset_index(drop=True)
    st.dataframe(ap, use_container_width=True, height=360)
    telecharger_df(df_probs, "probabilites_P_mot_topic_complet.csv")

    st.subheader("Visualisation PyLDAvis")
    st.markdown("Slide to adjust relevance metric: λ = 1")
    try:
        vis = preparer_pyldavis_depuis_sklearn(lda, X_counts, vect)
        html = pyLDAvis.prepared_data_to_html(vis)
        st.components.v1.html(html, height=800, scrolling=True)
        st.download_button(
            label="Télécharger la visualisation PyLDAvis (HTML)",
            data=html.encode("utf-8"),
            file_name="pyldavis_lda.html",
            mime="text/html",
            use_container_width=True
        )
        st.markdown(
            """
            <div style="line-height:1.6; margin-top:0.5rem;">
              <div>
                <span style="display:inline-block;width:12px;height:12px;background:#1f77b4;margin-right:8px;border:1px solid #333;"></span>
                Overall term frequency (couleur bleue) : fréquence globale du terme dans l’ensemble du corpus.
              </div>
              <div>
                <span style="display:inline-block;width:12px;height:12px;background:#d62728;margin-right:8px;border:1px solid #333;"></span>
                Estimated term frequency within the selected topic (couleur rouge) : fréquence estimée du terme dans le topic sélectionné.
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            "Le curseur de pertinence λ règle la manière de classer les termes dans le panneau de droite. "
            "Avec λ = 1, l’ordonnancement privilégie la probabilité P(mot | topic). "
            "En abaissant λ, on met davantage l’accent sur la spécificité des termes au topic par rapport au corpus."
        )
    except Exception as e:
        st.warning(f"Impossible d’afficher PyLDAvis : {e}")

    duree = time.time() - debut
    st.caption(f"Terminé en {duree:.1f} secondes.")



