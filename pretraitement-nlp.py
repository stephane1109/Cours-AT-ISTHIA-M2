# pip install spacy streamlit nltk
# python -m spacy download fr_core_news_md

# python -m streamlit run main.py

import streamlit as st
import spacy
import re
from nltk.stem.snowball import FrenchStemmer

# Initialisation
nlp = spacy.load("fr_core_news_md")
stemmer = FrenchStemmer()

# Fonction d'affichage explicative
def afficher_tags_pos():
    st.markdown("### Liste des étiquettes grammaticales SpaCy (POS)")
    st.markdown("""
| Étiquette | Signification |
|----------|---------------|
| ADJ      | adjectif |
| ADP      | préposition |
| ADV      | adverbe |
| AUX      | auxiliaire |
| CCONJ    | conjonction de coordination |
| DET      | déterminant |
| INTJ     | interjection |
| NOUN     | nom commun |
| NUM      | nombre |
| PART     | particule |
| PRON     | pronom |
| PROPN    | nom propre |
| PUNCT    | ponctuation |
| SCONJ    | conjonction de subordination |
| SYM      | symbole |
| VERB     | verbe |
| X        | autre / inconnu |
""")

# Fonction de nettoyage
def nettoyer_texte(
    texte, options, lemmatisation=False, stemming=False,
    keep_case=True, remove_numbers=False, supprimer_contractes=False
):
    if not keep_case:
        texte = texte.lower()

    if remove_numbers:
        texte = re.sub(r"\d+", " ", texte)

    texte = re.sub(r"[^\w\s]", " ", texte)  # suppression ponctuation
    doc = nlp(texte)
    mots_utiles = []

    for token in doc:
        if token.is_space:
            continue

        mot = token.lemma_ if lemmatisation else token.text
        if stemming:
            mot = stemmer.stem(token.text)

        # Supprimer les formes contractées isolées (ex : "c", "j", "l"...)
        if supprimer_contractes and (mot in {"c", "j", "l", "d", "t", "m", "n", "s", "qu"} or len(mot) == 1):
            continue

        # Suppressions selon POS
        if ("Supprimer les déterminants" in options and token.pos_ == "DET") or \
           ("Supprimer les pronoms" in options and token.pos_ == "PRON") or \
           ("Supprimer les prépositions" in options and token.pos_ == "ADP") or \
           ("Supprimer les conjonctions" in options and token.pos_ in ["CCONJ", "SCONJ"]) or \
           ("Supprimer les adverbes" in options and token.pos_ == "ADV") or \
           ("Supprimer tous les verbes" in options and token.pos_ in ["VERB", "AUX"]):
            continue

        mots_utiles.append(mot)

    return mots_utiles


# TITRE PRINCIPAL
st.title("Prétraitement linguistique d’un texte (français)")

st.markdown("""
Cette application permet de **prétraiter un fichier texte** (.txt) : suppression de certains mots grammaticaux, lemmatisation, stemming, minuscule, chiffres, etc.
""")

# UPLOADER LE FICHIER
fichier = st.file_uploader("Importer un fichier texte (.txt)", type=["txt"], key="fichier_txt")

if fichier is not None:
    texte_brut = fichier.read().decode("utf-8")
    st.subheader("Aperçu du texte original")
    st.text(texte_brut[:1000] + "...")

    # Options de traitement (Sidebar)
    st.sidebar.header("🛠️ Options de nettoyage")
    opt_dets = st.sidebar.checkbox("Supprimer les déterminants (le, la, un...)", value=True)
    opt_pron = st.sidebar.checkbox("Supprimer les pronoms (je, il, nous...)", value=True)
    opt_prep = st.sidebar.checkbox("Supprimer les prépositions (à, de, avec...)", value=True)
    opt_conj = st.sidebar.checkbox("Supprimer les conjonctions (et, mais...)", value=True)
    opt_advb = st.sidebar.checkbox("Supprimer les adverbes (bien, déjà...)", value=False)
    opt_verb = st.sidebar.checkbox("Supprimer tous les verbes", value=False)
    opt_chiffres = st.sidebar.checkbox("Supprimer les chiffres", value=True)
    opt_minuscule = st.sidebar.checkbox("Mettre le texte en minuscule", value=True)
    opt_contractes = st.sidebar.checkbox("Supprimer les formes contractées isolées (c, j, l...)", value=True)

    lemmatisation = st.sidebar.checkbox("Activer la lemmatisation (SpaCy)", value=True)
    stemming = st.sidebar.checkbox("Activer le stemming (Snowball)", value=False)

    # Construction de la liste des options cochées
    choix_options = [opt for opt, actif in zip([
        "Supprimer les déterminants",
        "Supprimer les pronoms",
        "Supprimer les prépositions",
        "Supprimer les conjonctions",
        "Supprimer les adverbes",
        "Supprimer tous les verbes"
    ], [opt_dets, opt_pron, opt_prep, opt_conj, opt_advb, opt_verb]) if actif]

    # Lancer le nettoyage
    if st.button("Lancer le prétraitement"):
        mots_nettoyes = nettoyer_texte(
            texte_brut,
            choix_options,
            lemmatisation=lemmatisation,
            stemming=stemming,
            keep_case=not opt_minuscule,
            remove_numbers=opt_chiffres,
            supprimer_contractes=opt_contractes
        )
        texte_final = " ".join(mots_nettoyes)

        st.subheader("Texte après prétraitement")
        st.text(texte_final[:1000] + "...")

        # Bouton de téléchargement
        st.download_button(
            label="Télécharger le texte nettoyé",
            data=texte_final.encode("utf-8"),
            file_name="texte_nettoye.txt",
            mime="text/plain"
        )

        st.markdown(f"**Nombre de mots après traitement :** {len(mots_nettoyes)}")

# Afficher les étiquettes grammaticales SpaCy
st.divider()
afficher_tags_pos()
