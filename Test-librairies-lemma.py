# pip install streamlit matplotlib pandas numpy
# pip install wordcloud
# pip install nltk spacy
# python -m spacy download fr_core_news_sm
# python -m spacy download fr_core_news_md

# Lancer : python -m streamlit run main.py


# app_lemmatisation_fr.py
# -------------------------------------------------------------
# Comparateur de racinisation / lemmatisation en français.
#
# Méthodes incluses :
# - Porter Stemmer (NLTK) — stemming, conçu pour l’anglais (pédagogique)
# - Snowball Stemmer (NLTK, "french") — stemming adapté au français
# - WordNet Lemmatizer (NLTK) — lemmatisation surtout pour l’anglais (pédagogique en FR)
# - spaCy fr_core_news_sm — lemmatisation française (petit modèle)
# - spaCy fr_core_news_md — lemmatisation française (modèle moyen)
#
# Fonctionnement :
# - Saisie d’un texte en français
# - Tokenisation regex + retrait stopwords NLTK (optionnel)
# - Pour chaque méthode : DataFrame (token_original, forme_transformee) alignée par occurrence
# - Tableau comparatif final croisant toutes les méthodes sélectionnées
# -------------------------------------------------------------

import re
import pandas as pd
import streamlit as st
import nltk     # NLTK : stopwords et stemming
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer, WordNetLemmatizer
import spacy

# Téléchargements de NLTK nécessaires
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ============================
# Utilitaires texte
# ============================

def tokeniser_fr(texte: str):
    """Tokenise en minuscules, en conservant lettres accentuées et underscore."""
    return re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ_]+", texte.lower())

@st.cache_resource(show_spinner=False)
def charger_stopwords_fr():
    """Stopwords français NLTK, avec ajustement pour 'aujourd'hui'."""
    sw = set(w.lower() for w in stopwords.words("french"))
    sw.update({"aujourd", "hui"})
    return sw

def filtrer_tokens(tokens, stopwords_fr=None, longueur_min=2):
    """Retire les stopwords si fournis et les tokens trop courts."""
    if stopwords_fr is None:
        return [t for t in tokens if len(t) >= longueur_min]
    return [t for t in tokens if (t not in stopwords_fr and len(t) >= longueur_min)]

def dataframe_mapping(tokens, formes):
    """Construit une DataFrame par occurrence : token_original → forme_transformee."""
    n = min(len(tokens), len(formes))
    return pd.DataFrame({"token_original": tokens[:n], "forme_transformee": formes[:n]})

# ============================
# Méthodes
# ============================

def appliquer_porter(tokens):
    """Stemming conçu pour l’anglais ; en français, usage pédagogique."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def appliquer_snowball_fr(tokens):
    """Stemming français (suppression de suffixes)."""
    stemmer = SnowballStemmer("french")
    return [stemmer.stem(t) for t in tokens]

def appliquer_wordnet_lemmatizer(tokens):
    """Lemmatisation surtout pour l’anglais ; très limité en français."""
    lemm = WordNetLemmatizer()
    return [lemm.lemmatize(t) for t in tokens]

@st.cache_resource(show_spinner=True)
def charger_modele_spacy(nom: str):
    """Charge un modèle spaCy donné."""
    return spacy.load(nom)

def appliquer_spacy_par_token(tokens, nom_modele="fr_core_news_sm"):
    """
    On utilise spaCy sur chaque mot (token) séparément pour être sûr d’obtenir exactement le même nombre
    de mots en sortie qu’en entrée. Cela évite les décalages liés à la façon dont spaCy coupe le texte.
    """
    nlp = charger_modele_spacy(nom_modele)
    lemmes = []
    for t in tokens:
        doc = nlp(t)
        if len(doc) > 0 and doc[0].lemma_:
            lemmes.append(doc[0].lemma_.lower())
        else:
            lemmes.append(t)
    return lemmes

# ==========================
# Interface Streamlit
# ==========================

st.set_page_config(page_title="Comparateur lemmatisation et Stemming - FR", layout="centered")
st.title("Comparaison des librairies Python de lemmatisation et de Stemming pour le français")

st.markdown(
    "Le stemming tronque les mots pour obtenir une racine commune (Porter, Snowball). "
    "La lemmatisation renvoie la forme canonique d’un mot (WordNet, spaCy). "
    "Ici, on compare aussi spaCy « sm » et « md » sur le même texte pour visualiser les différences éventuelles."
)

# Paragraphe explicatif
st.markdown("""
### 1. Stemming (racinisation)
Le stemming coupe les mots en supprimant les suffixes ou préfixes selon des règles simples.  
Le résultat n’est pas forcément un mot correct en français.  

**Exemple avec le stemmer Snowball :**
- *mangeaient* → **mang**  
- *étudiants* → **etudi**  

---

### 2. Lemmatisation
La lemmatisation repose sur un dictionnaire linguistique ou un modèle (spaCy, UDPipe, etc.).  
Le résultat est la forme canonique du mot (lemme), celle qu’on trouve dans un dictionnaire.  

**Exemple avec spaCy (français) :**
- *mangeaient* → **manger** (infinitif du verbe)  
- *étudiants* → **étudiant** (nom au singulier)

---

### Remarque sur l'approche par dictionnaire (LEFFF / IRaMuTeQ)
L'approche par dictionnaire quant à elle **ne fait pas d'erreur sur les mots connus**, mais **peut en produire sur les mots non listés**.  
Dans la pratique, des ressources comme **LEFFF** ou le dictionnaire intégré d’**IRaMuTeQ** donnent souvent d’excellents résultats pour les formes couvertes.

---
""")

st.markdown("""### 3. Votre texte :""")
st.markdown(
    "Saisissez un court texte (environ 5 phrases). "
    "L’application appliquera les différentes méthodes de lemmatisation "
    "ou de stemming et affiche les résultats pour comparaison."

)

texte = st.text_area(
    "Entrez votre texte (≈ 5 phrases)",
    height=180,
    value="Le surtourisme fragilise certains territoires et modifie la vie des habitants. "
          "La psychiatrie étudiante est un enjeu croissant avec la solitude et l’anxiété. "
          "La gastronomie et la culture attirent des visiteurs et soutiennent l’économie locale. "
          "Les influenceurs orientent parfois la fréquentation vers des sites fragiles. "
          "Les politiques publiques cherchent un équilibre entre attractivité et préservation."
)

retirer_stop = st.checkbox("Retirer les stopwords (NLTK)", value=True)

tokens = tokeniser_fr(texte)
tokens_filtres = filtrer_tokens(tokens, charger_stopwords_fr() if retirer_stop else None, longueur_min=2)

st.markdown("Aperçu des tokens normalisés")
st.write(tokens_filtres[:40])
st.write(f"Nombre de tokens après filtrage : {len(tokens_filtres)}")

st.markdown("---")
st.subheader("Sélection des méthodes")
col1, col2 = st.columns(2)
with col1:
    sel_porter = st.checkbox("NLTK - Porter Stemmer — stemming", value=True)
    sel_snowball = st.checkbox("NLTK - Snowball Stemmer (français) — stemming", value=True)
with col2:
    sel_wordnet = st.checkbox("WordNet Lemmatizer (NLTK) — lemmatisation surtout anglais", value=False)
    sel_spacy_sm = st.checkbox("spaCy — fr_core_news_sm (modèle small)", value=True)
    sel_spacy_md = st.checkbox("spaCy — fr_core_news_md (modèle medium)", value=True)

st.markdown("---")
st.subheader("Résultats par méthode")

dfs = []  # pour construire le comparatif final

if sel_porter:
    formes = appliquer_porter(tokens_filtres)
    df_porter = dataframe_mapping(tokens_filtres, formes)
    st.markdown("Librairie NLTK — Porter Stemmer : méthode de stemming conçue pour l’anglais")
    st.dataframe(df_porter, use_container_width=True)
    dfs.append(("Porter", df_porter.rename(columns={"forme_transformee": "Porter"})))

if sel_snowball:
    formes = appliquer_snowball_fr(tokens_filtres)
    df_snow = dataframe_mapping(tokens_filtres, formes)
    st.markdown("Librairie NLTK — Snowball Stemmer (français) : méthode de stemming adaptée au français, fondée sur la suppression de suffixes.")
    st.dataframe(df_snow, use_container_width=True)
    dfs.append(("SnowballFR", df_snow.rename(columns={"forme_transformee": "SnowballFR"})))

if sel_wordnet:
    formes = appliquer_wordnet_lemmatizer(tokens_filtres)
    df_wn = dataframe_mapping(tokens_filtres, formes)
    st.markdown("WordNet Lemmatizer : lemmatisation basée sur WordNet, surtout pour l’anglais")
    st.dataframe(df_wn, use_container_width=True)
    dfs.append(("WordNet", df_wn.rename(columns={"forme_transformee": "WordNet"})))

if sel_spacy_sm:
    try:
        formes = appliquer_spacy_par_token(tokens_filtres, "fr_core_news_sm")
        df_sm = dataframe_mapping(tokens_filtres, formes)
        st.markdown("spaCy (sm) — spaCy fr_core_news_sm : petit modèle français, plus léger et rapide ; lemmatisation basée sur règles/statistiques.")
        st.dataframe(df_sm, use_container_width=True)
        dfs.append(("spaCy_sm", df_sm.rename(columns={"forme_transformee": "spaCy_sm"})))
    except Exception as e:
        st.error(f"spaCy (sm) n’a pas pu être exécuté : {e}")

if sel_spacy_md:
    try:
        formes = appliquer_spacy_par_token(tokens_filtres, "fr_core_news_md")
        df_md = dataframe_mapping(tokens_filtres, formes)
        st.markdown("spaCy (md) — spaCy fr_core_news_md : modèle moyen, plus riche (vecteurs) ; peut améliorer certaines lemmatisations et analyses.")
        st.dataframe(df_md, use_container_width=True)
        dfs.append(("spaCy_md", df_md.rename(columns={"forme_transformee": "spaCy_md"})))
    except Exception as e:
        st.error(f"spaCy (md) n’a pas pu être exécuté : {e}")

st.markdown("---")
st.subheader("Tableau comparatif final")

if len(dfs) > 0:
    df_ref = pd.DataFrame({"token_original": tokens_filtres})
    for nom, d in dfs:
        df_ref = df_ref.merge(d, on="token_original", how="left")
    st.dataframe(df_ref, use_container_width=True)
else:
    st.info("Aucune méthode sélectionnée.")
