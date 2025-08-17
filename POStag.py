# pip install streamlit pandas spacy
# python -m spacy download fr_core_news_md

# Lancer : python -m streamlit run main.py

import re
import pandas as pd
import streamlit as st
import spacy

# ============================
# Utilitaires texte
# ============================

def tokeniser_fr(texte: str):
    """
    Tokenise en minuscules avec une regex simple, en conservant les lettres accentuées et l’underscore.
    Cette tokenisation ne dépend pas de spaCy.
    """
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_]+", texte.lower())

def dataframe_mapping(tokens, formes, col_tokens="token", col_formes="lemme"):
    """
    Construit une DataFrame alignée 1:1 entre la liste de tokens et la liste de formes.
    Tronque à la longueur minimale commune par sécurité.
    """
    n = min(len(tokens), len(formes))
    return pd.DataFrame({col_tokens: tokens[:n], col_formes: formes[:n]})

# ============================
# spaCy (modèle MD uniquement)
# ============================

@st.cache_resource(show_spinner=True)
def charger_modele_spacy_md():
    """Charge le modèle spaCy français medium."""
    return spacy.load("fr_core_news_md")

def extraire_tokens_spacy(texte: str, retirer_stop: bool = True):
    """
    Retourne une liste de tokens (minuscules) selon spaCy, avec retrait éventuel des stopwords spaCy.
    Cette liste sert d’entrée au mode « par tokens ».
    """
    nlp = charger_modele_spacy_md()
    doc = nlp(texte)
    tokens = []
    for t in doc:
        if t.is_space:
            continue
        if not t.is_alpha:
            continue
        if retirer_stop and t.is_stop:
            continue
        tokens.append(t.text.lower())
    return tokens

def lemmatiser_par_token_spacy(tokens):
    """
    Lemmatisation hors contexte : applique spaCy MD sur chaque token isolé.
    La sortie a la même longueur que l’entrée.
    """
    nlp = charger_modele_spacy_md()
    lemmes = []
    for t in tokens:
        doc = nlp(t)
        if len(doc) > 0 and doc[0].lemma_:
            lemmes.append(doc[0].lemma_.lower())
        else:
            lemmes.append(t)
    return lemmes

def analyser_par_token_spacy(tokens):
    """
    Analyse morpho-syntaxique par token isolé.
    Renvoie une DataFrame avec token, pos, morph, lemma, is_stopword.
    Attention : sans contexte, le POS et le lemme peuvent être moins fiables.
    """
    nlp = charger_modele_spacy_md()
    rows = []
    for t in tokens:
        doc = nlp(t)
        if len(doc) == 0:
            rows.append({"token": t, "pos": None, "morph": {}, "lemma": t, "is_stopword": False})
            continue
        tok = doc[0]
        rows.append({
            "token": t,
            "pos": tok.pos_,
            "morph": tok.morph.to_json(),
            "lemma": tok.lemma_.lower() if tok.lemma_ else t,
            "is_stopword": bool(tok.is_stop)
        })
    return pd.DataFrame(rows)

# ============================
# Affichage aligné mots + étiquettes (visualisation par phrases)
# ============================

def html_mots_etiquettes(doc, label_mode: str = "POS", afficher_stop: bool = True):
    """
    Rend une phrase sur une ligne avec, sous chaque mot, l’étiquette choisie.
    label_mode ∈ {"POS", "LEMMA", "POS+LEMMA"}.
    Cette visualisation s’appuie sur la segmentation de spaCy mais n’affecte pas le traitement « par tokens ».
    """
    css = """
    <style>
    .ligne-phrase { margin: 0.4rem 0 0.2rem 0; white-space: nowrap; overflow-x: auto; }
    .grille-tokens { display: inline-flex; gap: 12px; align-items: flex-start; }
    .token-bloc { display: inline-flex; flex-direction: column; align-items: center; }
    .mot { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
           font-size: 15px; padding: 2px 4px; border-bottom: 1px solid #bbb; }
    .etiquette { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                 font-size: 12px; color: #444; }
    .stop { opacity: 0.45; }
    </style>
    """
    blocs = []
    for t in doc:
        if t.is_space or not t.is_alpha:
            continue
        if not afficher_stop and t.is_stop:
            continue
        if label_mode == "POS":
            lab = t.pos_
        elif label_mode == "LEMMA":
            lab = t.lemma_.lower()
        else:
            lab = f"{t.pos_} | {t.lemma_.lower()}"
        cls_stop = " stop" if t.is_stop else ""
        blocs.append(
            f'<div class="token-bloc{cls_stop}"><div class="mot">{t.text}</div>'
            f'<div class="etiquette">{lab}</div></div>'
        )
    html = css + f'<div class="ligne-phrase"><div class="grille-tokens">{"".join(blocs)}</div></div>'
    return html

def rendre_alignement_par_phrases(texte: str, label_mode: str = "POS", afficher_stop: bool = True):
    """
    Génère un rendu HTML pour chaque phrase du texte avec mots en ligne et étiquette dessous.
    Purement visuel ; ne change rien au traitement par tokens.
    """
    nlp = charger_modele_spacy_md()
    doc = nlp(texte)
    out = []
    for sent in doc.sents:
        out.append(html_mots_etiquettes(sent, label_mode=label_mode, afficher_stop=afficher_stop))
    return "\n".join(out)

# ==========================
# Interface Streamlit
# ==========================

st.set_page_config(page_title="Lemmatisation et POS-tagging — spaCy MD", layout="centered")
st.title("Lemmatisation et POS-tagging (spaCy, modèle medium)")

st.markdown(
    "### Qu’est-ce que le POS-tagging ?\n"
    "Le **POS-tagging** (Part-of-Speech tagging) ou **étiquetage morpho-syntaxique** consiste à attribuer "
    "à chaque mot de la phrase une **catégorie grammaticale** (nom, verbe, adjectif, etc...) "
    "et des traits morphologiques (genre, nombre, temps, etc.).\n\n"
    "Dans spaCy, le POS-tagging s’appuie sur un modèle entraîné qui analyse le **contexte** de chaque mot dans la phrase. "
    "Cela permet de distinguer des cas ambigus :\n"
    "- **« recherche »** → peut être un **nom** (« ma recherche ») ou un **verbe** (« je recherche »).\n\n"
    "Cette étape est fondamentale car le lemme dépend du POS. Ainsi, le POS-tagging améliore la précision "
    "de la **lemmatisation** et prépare l’**analyse sémantique**."
)

st.markdown(
    "Le POS-tagging ouvre vers une analyse sémantique plus fiable."
)

st.markdown("### Saisissez votre texte :")
texte = st.text_area(
    "Texte (≈ 5 phrases)",
    height=180,
    value="La recherche progresse vite dans ce laboratoire. Je recherche des sources fiables. "
          "Cet avocat plaide au tribunal, tandis que l’avocat bien mûr se tartine sur du pain. "
          "Le surtourisme fragilise certains territoires et modifie la vie des habitants."
)

retirer_stop = st.checkbox("Retirer les stopwords selon spaCy (pour le traitement par tokens)", value=True)

# Traitement par tokens avec spaCy pour le filtrage des stopwords
tokens_spacy = extraire_tokens_spacy(texte, retirer_stop=retirer_stop)

st.subheader("Aperçu des tokens (spaCy, après éventuel retrait des stopwords)")
st.write(tokens_spacy[:40])
st.write(f"Nombre de tokens conservés : {len(tokens_spacy)}")

st.markdown("---")
st.subheader("Tableau token → lemme (hors contexte, spaCy MD)")

try:
    lemmes = lemmatiser_par_token_spacy(tokens_spacy)
    df_lemmas = dataframe_mapping(tokens_spacy, lemmes)
    st.dataframe(df_lemmas, use_container_width=True)
except Exception as e:
    st.error(f"Erreur lors de la lemmatisation par tokens : {e}")

st.markdown("---")
st.subheader("Analyse morpho-syntaxique par token")

afficher_pos = st.checkbox("Afficher POS et traits morphologiques (par token)", value=True)
if afficher_pos:
    try:
        df_pos = analyser_par_token_spacy(tokens_spacy)
        st.dataframe(df_pos, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur lors de l’analyse POS par tokens : {e}")

st.markdown("---")
st.subheader("Affichage aligné par phrase (visualisation)")

label_mode = st.selectbox("Étiquette à afficher", ["POS", "LEMMA", "POS+LEMMA"], index=0)
afficher_stop_visu = st.checkbox("Afficher aussi les stopwords dans la visualisation alignée", value=False)

try:
    html_visu = rendre_alignement_par_phrases(texte, label_mode=label_mode, afficher_stop=afficher_stop_visu)
    st.markdown(html_visu, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Erreur de rendu aligné : {e}")

# Export CSV
st.markdown("---")
st.subheader("Export des résultats")
col1, col2 = st.columns(2)
with col1:
    if 'df_lemmas' in locals():
        st.download_button(
            "Télécharger token→lemme (CSV)",
            data=df_lemmas.to_csv(index=False).encode("utf-8"),
            file_name="tokens_lemmes_spacy_md.csv",
            mime="text/csv"
        )
with col2:
    if afficher_pos and 'df_pos' in locals():
        st.download_button(
            "Télécharger POS par token (CSV)",
            data=df_pos.to_csv(index=False).encode("utf-8"),
            file_name="pos_par_tokens_spacy_md.csv",
            mime="text/csv"
        )
