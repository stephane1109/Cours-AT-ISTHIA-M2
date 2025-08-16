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
    """Tokenise en minuscules, en conservant lettres accentuées et underscore.
    (Conservée pour compatibilité ; le filtrage des stopwords est fait via spaCy.)
    """
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_]+", texte.lower())

def dataframe_mapping(tokens, formes):
    """Construit une DataFrame par occurrence : token_original → forme_transformee."""
    n = min(len(tokens), len(formes))
    return pd.DataFrame({"token_original": tokens[:n], "forme_transformee": formes[:n]})

# ============================
# spaCy (modèle MD uniquement)
# ============================

@st.cache_resource(show_spinner=True)
def charger_modele_spacy_md():
    """Charge le modèle spaCy français medium."""
    return spacy.load("fr_core_news_md")

def extraire_tokens_spacy(texte: str, retirer_stop: bool = True):
    """Retourne une liste de tokens (minuscules) selon spaCy, en retirant éventuellement les stopwords spaCy."""
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

def appliquer_spacy_md_par_token(tokens):
    """
    Applique spaCy MD sur chaque token isolément pour garantir une sortie
    de même longueur que l’entrée.
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

def _tokens_spacy_depuis_phrase_md(texte: str, retirer_stop: bool):
    """Extrait les tokens depuis la segmentation spaCy MD sur la phrase complète, avec filtre spaCy (is_stop) si demandé."""
    return extraire_tokens_spacy(texte, retirer_stop=retirer_stop)

def appliquer_spacy_md_par_phrase(texte: str, retirer_stop: bool = True):
    """
    Analyse la phrase complète avec spaCy MD, puis applique la lemmatisation par token
    pour conserver un alignement 1:1 avec la liste de tokens issue de la segmentation spaCy.
    """
    nlp = charger_modele_spacy_md()
    tokens_spacy = _tokens_spacy_depuis_phrase_md(texte, retirer_stop)
    lemmes = []
    for tok in tokens_spacy:
        doc_t = nlp(tok)
        if len(doc_t) > 0 and doc_t[0].lemma_:
            lemmes.append(doc_t[0].lemma_.lower())
        else:
            lemmes.append(tok)
    return tokens_spacy, lemmes

def analyser_spacy_md_pos(texte: str, filtrer_stop: bool = False):
    """
    Analyse morpho-syntaxique via spaCy MD sur la phrase saisie.
    Renvoie token, POS, traits morphologiques et lemme pour chaque token.
    Filtrage de stopwords via spaCy (t.is_stop).
    """
    nlp = charger_modele_spacy_md()
    doc = nlp(texte)
    rows = []
    for t in doc:
        if t.is_space or not t.is_alpha:
            continue
        if filtrer_stop and t.is_stop:
            continue
        rows.append({
            "token": t.text,
            "pos": t.pos_,
            "morph": t.morph.to_json(),
            "lemma": t.lemma_,
            "is_stop": bool(t.is_stop)
        })
    return pd.DataFrame(rows)

# ============================
# Rendu aligné mots + étiquettes
# ============================

def html_mots_etiquettes(doc, label_mode: str = "POS", afficher_stop: bool = True):
    """
    Rend une phrase sur une ligne avec, sous chaque mot, l'étiquette choisie.
    label_mode ∈ {"POS", "LEMMA", "POS+LEMMA"}
    """
    # Style minimaliste pour l’alignement
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
        blocs.append(f'<div class="token-bloc{cls_stop}"><div class="mot">{t.text}</div><div class="etiquette">{lab}</div></div>')
    html = css + f'<div class="ligne-phrase"><div class="grille-tokens">{"".join(blocs)}</div></div>'
    return html

def rendre_alignement_par_phrases(texte: str, label_mode: str = "POS", afficher_stop: bool = True):
    """
    Génère un rendu HTML pour chaque phrase du texte avec mots en ligne et étiquette dessous.
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

st.set_page_config(page_title="Lemmatisation française — spaCy MD", layout="centered")
st.title("Lemmatisation et étiquetage morpho-syntaxique (spaCy, modèle medium)")

st.markdown(
    "La normalisation par lemmatisation réduit la variabilité des formes (pluriels, conjugaisons, dérivations) "
    "afin de mieux comparer et analyser les textes."
)

st.markdown(
    "Le **POS-tagging** (étiquetage morpho-syntaxique) assigne à chaque mot une **catégorie grammaticale** "
    "(nom, verbe, adjectif, etc.). Cette information permet de choisir le **bon lemme** selon le rôle du mot "
    "dans la phrase et d’aborder l’**analyse sémantique** plus finement. "
    "Par exemple, le mot **« avocat »** peut désigner une **profession** (NOUN → lemme « avocat ») "
    "ou le **fruit** (NOUN → lemme « avocat »), la désambiguïsation se fait par le **contexte** ; "
    "de même, **« recherche »** peut être un **nom** (« ma recherche ») ou le **verbe** « rechercher » "
    "(selon le contexte et le POS, le lemme diffère)."
)

st.markdown("### Saisissez votre texte")
texte = st.text_area(
    "Texte (≈ 5 phrases)",
    height=180,
    value="La recherche progresse vite dans ce laboratoire. Je dois rechercher des sources fiables. "
          "Cet avocat plaide au tribunal, tandis que l’avocat bien mûr se tartine sur du pain. "
          "Le surtourisme fragilise certains territoires et modifie la vie des habitants."
)

retirer_stop = st.checkbox("Retirer les stopwords selon spaCy", value=True)

# Préparation « mode par token » à partir de spaCy (filtre spaCy)
tokens_filtres = extraire_tokens_spacy(texte, retirer_stop=retirer_stop)

st.markdown("Aperçu des tokens normalisés (mode par token, stopwords gérés par spaCy)")
st.write(tokens_filtres[:40])
st.write(f"Nombre de tokens après filtrage : {len(tokens_filtres)}")

st.markdown("---")
st.subheader("Mode d’application de la lemmatisation (spaCy MD)")
mode_spacy = st.radio(
    "Choisir le mode",
    ["Par token (même longueur que l’entrée)", "Par phrase (segmentation spaCy)"],
    index=0
)

st.markdown("---")
st.subheader("Résultats de lemmatisation (spaCy MD)")

if mode_spacy.startswith("Par token"):
    try:
        formes = appliquer_spacy_md_par_token(tokens_filtres)
        df_md = dataframe_mapping(tokens_filtres, formes)
        st.dataframe(df_md, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur spaCy MD (par token) : {e}")
else:
    try:
        toks_spacy, formes = appliquer_spacy_md_par_phrase(texte, retirer_stop=retirer_stop)
        df_md = dataframe_mapping(toks_spacy, formes)
        st.caption("Tokens issus de la segmentation spaCy MD, filtrés avec les stopwords spaCy.")
        st.dataframe(df_md, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur spaCy MD (par phrase) : {e}")

st.markdown("---")
st.subheader("Analyse POS et traits morphologiques (spaCy MD) sur la phrase saisie")
afficher_pos = st.checkbox("Afficher POS, traits morphologiques et lemmes", value=True)
if afficher_pos:
    try:
        df_pos = analyser_spacy_md_pos(texte, filtrer_stop=False)
        st.dataframe(df_pos, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur d’analyse POS spaCy MD : {e}")

st.markdown("---")
st.subheader("Affichage aligné : mots sur une ligne, étiquette dessous")
label_mode = st.selectbox(
    "Étiquette à afficher",
    options=["POS", "LEMMA", "POS+LEMMA"],
    index=0
)
afficher_stop_visu = st.checkbox("Afficher aussi les stopwords dans la visualisation alignée", value=False)

try:
    html_visu = rendre_alignement_par_phrases(texte, label_mode=label_mode, afficher_stop=afficher_stop_visu)
    st.markdown(html_visu, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Erreur de rendu aligné : {e}")

