# python -m streamlit run main.py
# pip install streamlit pandas matplotlib worldcloud
# python -m spacy download fr_core_news_md

# Remarque : les stopwords sont appliqués partout (fenêtres et comptages) quand la case est cochée.

# -*- coding: utf-8 -*-
# Application Streamlit : Cooccurrences par fréquence autour d’un mot pivot
# - Un seul mode : cooccurrences par fréquence
# - Fenêtre : Mots (±k), Phrase, Paragraphe, Document
# - Stopwords optionnels (spaCy) + option pour exclure les nombres (0-9)
# - Le pivot n’est jamais filtré (ni par stopwords, ni par l’option d’exclusion des nombres)
# - Tableau pandas : cooccurrent, pos, frequence, fenetres_ensemble, loglike
# - Export CSV, nuage de mots sécurisé, explications intégrées (help=) et onglet Aide
# - Paramètres sous le champ texte, aucune barre latérale

# ================================
# IMPORTS
# ================================
import io
import re
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from wordcloud import WordCloud
import spacy

# ================================
# CHARGEMENT SPACY
# ================================
try:
    nlp = spacy.load("fr_core_news_md")
except Exception:
    raise RuntimeError(
        "Le modèle spaCy 'fr_core_news_md' n'est pas installé.\n"
        "Installez-le avec :\n"
        "pip install -U spacy && python -m spacy download fr_core_news_md"
    )

# ================================
# STOPWORDS
# ================================
def construire_stopwords(personnalises: str, appliquer_stop: bool):
    """
    Construit l’ensemble des stopwords à appliquer.
    - Si appliquer_stop == False : retourne un ensemble vide (aucun filtrage).
    - Si appliquer_stop == True : utilise les stopwords spaCy + ceux fournis par l’utilisateur.
    Aucun ajout manuel par défaut.
    """
    if not appliquer_stop:
        return set()
    sw = set(nlp.Defaults.stop_words)
    if personnalises.strip():
        extras = [w.strip().lower() for w in personnalises.split(",") if w.strip()]
        sw.update(extras)
    return sw

# ================================
# SEGMENTATION PARAGRAPHES
# ================================
def segmenter_paragraphes(texte: str):
    """Paragraphe = bloc séparé par au moins une ligne vide."""
    norm = texte.replace("\r\n", "\n").replace("\r", "\n")
    blocs = re.split(r"\n\s*\n+", norm)
    return [b.strip() for b in blocs if b.strip()]

# ================================
# FILTRAGE TOKEN
# ================================
def garder_token(token, stopset, pivot: str, exclure_nombres: bool):
    """
    Règle de filtrage uniforme pour toutes les fenêtres et pour l’étiquetage POS :
    - on garde uniquement les tokens alphanumériques,
    - si exclure_nombres == True : on retire les tokens composés uniquement de chiffres,
    - on retire les stopwords si cochés,
    - le pivot N’EST JAMAIS filtré.
    """
    txt = token.text
    if not txt.isalnum():
        return False
    w = txt.lower()
    if w == pivot:
        return True
    if exclure_nombres and w.isdigit():
        return False
    if w in stopset:
        return False
    return True

# ================================
# FENÊTRES DE CONTEXTE (filtrage appliqué, pivot jamais filtré)
# ================================
def fenetres_mots(doc, pivot: str, k: int, stopset, exclure_nombres: bool):
    """Fenêtres ±k centrées sur chaque occurrence du pivot, après filtrage uniforme."""
    seq = []
    for t in doc:
        if garder_token(t, stopset, pivot, exclure_nombres):
            seq.append(t.text.lower())
    indices = [i for i, w in enumerate(seq) if w == pivot]
    fenetres = []
    for idx in indices:
        d = max(0, idx - k)
        f = min(len(seq), idx + k + 1)
        fen = set(seq[d:f])
        if fen:
            fenetres.append(fen)
    return fenetres

def fenetres_phrases(doc, stopset, pivot, exclure_nombres: bool):
    """Fenêtres = phrases spaCy (., ?, !, … et règles du modèle), après filtrage uniforme."""
    fenetres = []
    for sent in doc.sents:
        mots = [t.text.lower() for t in sent if garder_token(t, stopset, pivot, exclure_nombres)]
        if mots:
            fenetres.append(set(mots))
    return fenetres

def fenetres_paragraphes(texte: str, stopset, pivot, exclure_nombres: bool):
    """Fenêtres = paragraphes (ligne vide comme séparateur), après filtrage uniforme."""
    fenetres = []
    for pa in segmenter_paragraphes(texte):
        d = nlp(pa)
        mots = [t.text.lower() for t in d if garder_token(t, stopset, pivot, exclure_nombres)]
        if mots:
            fenetres.append(set(mots))
    return fenetres

def fenetres_document(doc, stopset, pivot, exclure_nombres: bool):
    """Fenêtre = document entier, après filtrage uniforme."""
    mots = [t.text.lower() for t in doc if garder_token(t, stopset, pivot, exclure_nombres)]
    return [set(mots)] if mots else []

# ================================
# LOG-LIKELIHOOD (mêmes fenêtres que la fréquence)
# ================================
def compter_loglike_sur_fenetres(fenetres, pivot: str):
    """
    Calcule le log-likelihood sur des fenêtres (ensembles de mots).
    Retourne : scores (dict), T (nb fenêtres), F1 (fenêtres avec pivot), F12 (dict pivot&mot).
    """
    T = len(fenetres)
    if T == 0:
        return {}, 0, 0, {}
    F1 = 0
    F2 = Counter()
    F12 = Counter()
    for S in fenetres:
        contient_pivot = pivot in S
        if contient_pivot:
            F1 += 1
        for w in S:
            F2[w] += 1
            if contient_pivot and w != pivot:
                F12[w] += 1
    scores = {}
    for w, f2 in F2.items():
        if w == pivot:
            continue
        a = F12[w]
        b = F1 - a
        c = f2 - a
        d = T - a - b - c
        if any(x < 0 for x in (a, b, c, d)):
            continue
        def xlogx(x, y):
            if x == 0 or y == 0:
                return 0.0
            return x * math.log(x / y)
        row1 = a + b
        row2 = c + d
        col1 = a + c
        col2 = b + d
        total = row1 + row2
        if total == 0:
            continue
        E_a = row1 * col1 / total
        E_b = row1 * col2 / total
        E_c = row2 * col1 / total
        E_d = row2 * col2 / total
        ll = 2.0 * (xlogx(a, E_a) + xlogx(b, E_b) + xlogx(c, E_c) + xlogx(d, E_d))
        scores[w] = max(ll, 0.0)
    return scores, T, F1, dict(F12)

# ================================
# UTILITAIRES (POS, CSV, WordCloud, Aide)
# ================================
def etiqueter_pos_corpus(textes, stopset, pivot, exclure_nombres: bool):
    """
    POS majoritaire pour chaque mot conservé (filtrage uniforme, pivot jamais filtré).
    """
    pos_counts = defaultdict(Counter)
    for txt in textes:
        d = nlp(txt)
        for tok in d:
            if garder_token(tok, stopset, pivot, exclure_nombres):
                w = tok.text.lower()
                pos_counts[w][tok.pos_] += 1
    return {w: ctr.most_common(1)[0][0] for w, ctr in pos_counts.items()}

def generer_csv(df):
    """Flux CSV téléchargeable depuis un DataFrame."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def generer_wordcloud(freq_dict, titre):
    """Nuage de mots robuste (fréquences > 0, évite ZeroDivisionError)."""
    freq_pos = {w: int(f) for w, f in freq_dict.items() if f and f > 0}
    if not freq_pos:
        st.info("Nuage de mots non généré : aucune fréquence strictement positive.")
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq_pos)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(titre)
    st.pyplot(fig)

def texte_aide():
    """
    Renvoie un bloc Markdown expliquant clairement chaque paramètre et les découpages.
    """
    return (
        "### Aide et définitions\n\n"
        "**Mot pivot** : terme autour duquel on calcule les cooccurrences.\n\n"
        "**Fenêtre de contexte** :\n"
        "- **Mots (±k)** : on prend k mots avant et k mots après chaque occurrence du pivot. "
        "Le calcul se fait après filtrage (stopwords et/ou nombres si sélectionné).\n"
        "- **Phrase** : phrases détectées par spaCy (ponctuation terminale comme `.`, `?`, `!`, `…` "
        "et règles du modèle). Chaque phrase devient une fenêtre.\n"
        "- **Paragraphe** : blocs séparés par **au moins une ligne vide**. Chaque bloc devient une fenêtre.\n"
        "- **Document** : tout le texte est une seule fenêtre.\n\n"
        "**Stopwords (spaCy)** : mots-outils retirés si l’option est cochée. Vous pouvez en ajouter manuellement.\n\n"
        "**Exclure les nombres** : retire les tokens composés uniquement de chiffres (`'2024'`, `'123'`). "
        "Les formes alphanumériques comme `R2D2` sont conservées.\n\n"
        "**POS** : catégorie morpho-syntaxique (NOUN, VERB, ADJ, etc.), affichée comme information d’interprétation.\n\n"
        "**fenetres_ensemble** : nombre d’unités de contexte (selon la fenêtre choisie) contenant simultanément le pivot et le cooccurrent.\n\n"
        "**loglike** : mesure la force de l’association en comparant la cooccurrence observée à l’indépendance attendue, "
        "calculée sur les **mêmes fenêtres** que la fréquence.\n"
    )

def expliquer_loglike_md():
    """Texte explicatif du log-likelihood (sans formule)."""
    return (
        "### Comprendre le log-likelihood\n\n"
        "Le log-likelihood évalue à quel point la cooccurrence observée entre le pivot et un mot "
        "est plus (ou moins) fréquente que ce qu’on attendrait s’ils étaient indépendants. "
        "Le calcul se fait sur les mêmes fenêtres de contexte que la fréquence (±k, phrase, paragraphe, document). "
        "On compte les fenêtres contenant le pivot, le mot, et leur coprésence, puis on mesure l’écart à l’indépendance."
    )

# ================================
# INTERFACE STREAMLIT (sans barre latérale)
# ================================
st.set_page_config(page_title="Cooccurrences (fréquence)", layout="centered")
st.markdown("# Cooccurrences par fréquence autour d’un mot pivot")
st.caption("Paramétrez le texte et l’espace de cooccurrence. Toutes les explications sont disponibles dans l’onglet « Aide ».")

tab_texte, tab_resultats, tab_aide = st.tabs(["Texte et paramètres", "Résultats", "Aide"])

with tab_texte:
    uploaded = st.file_uploader("Charger un fichier texte (.txt)", type=["txt"], help="Chargez un fichier brut UTF-8. Le contenu sera analysé tel quel.")
    texte_libre = st.text_area("Ou collez votre texte ici", height=220, help="Vous pouvez coller un extrait ou un document entier. Si un fichier est chargé, il est prioritaire.")
    pivot = st.text_input("Mot pivot (obligatoire)", value="soleil", help="Le mot autour duquel on calcule les cooccurrences. Le pivot n’est jamais filtré.").strip().lower()
    fenetre = st.selectbox("Fenêtre de contexte", ["Mots (±k)", "Phrase", "Paragraphe", "Document"],
                           help="Définit l’unité de contexte : ±k mots, phrase spaCy, paragraphe (ligne vide), ou document entier.")
    k = 5
    if fenetre == "Mots (±k)":
        k = st.number_input("Taille de la fenêtre k (mots avant/après)", min_value=1, max_value=100, value=5, step=1,
                            help="Nombre de mots pris avant et après chaque occurrence du pivot (après filtrage).")
    appliquer_stop = st.checkbox("Appliquer les stopwords (spaCy)", value=True, help="Retire les mots-outils si coché. Vous pouvez en ajouter ci-dessous.")
    stop_persos = st.text_input("Stopwords supplémentaires (séparés par des virgules)", value="",
                                help="Exemple : 'ainsi, donc, toutefois'. Laissez vide si aucun.")
    exclure_nombres = st.checkbox("Exclure les nombres (0-9)", value=False, help="Retire les tokens constitués uniquement de chiffres (ex. 2024). Le pivot n’est jamais filtré.")
    top_n = st.number_input("Top N pour le nuage de mots", min_value=1, max_value=500, value=10, step=1,
                            help="Nombre de cooccurrents affichés dans le nuage.")
    generer_nuage = st.checkbox("Générer un nuage de mots (pondéré par la fréquence)", value=True,
                                help="Affiche un nuage pondéré par les fréquences de cooccurrence.")

    lancer = st.button("Lancer l’analyse")

with tab_resultats:
    st.write("Les résultats s’afficheront ici après lancement.")

with tab_aide:
    st.markdown(texte_aide())
    st.markdown(expliquer_loglike_md())

# ================================
# TRAITEMENT
# ================================
if lancer:
    if not pivot:
        st.error("Veuillez saisir un mot pivot.")
        st.stop()

    texte = uploaded.read().decode("utf-8", errors="ignore") if uploaded else texte_libre
    if not texte or not texte.strip():
        st.error("Veuillez fournir un texte.")
        st.stop()

    stopset = construire_stopwords(stop_persos, appliquer_stop)
    doc = nlp(texte)

    if fenetre == "Mots (±k)":
        fenetres = fenetres_mots(doc, pivot, k, stopset, exclure_nombres)
    elif fenetre == "Phrase":
        fenetres = fenetres_phrases(doc, stopset, pivot, exclure_nombres)
    elif fenetre == "Paragraphe":
        fenetres = fenetres_paragraphes(texte, stopset, pivot, exclure_nombres)
    else:
        fenetres = fenetres_document(doc, stopset, pivot, exclure_nombres)

    compteur = Counter()
    for S in fenetres:
        if pivot in S:
            for w in S:
                if w != pivot:
                    compteur[w] += 1

    scores, T, F1, F12 = compter_loglike_sur_fenetres(fenetres, pivot)

    if T == 0 or F1 == 0:
        st.warning("Aucune fenêtre valide contenant le pivot après filtrage. Vérifiez le texte, le pivot, les stopwords et l’option 'Exclure les nombres'.")

    pos_tags = etiqueter_pos_corpus([texte], stopset, pivot, exclure_nombres)

    coocs = sorted(set(compteur.keys()) | set(scores.keys()))
    lignes = []
    for w in coocs:
        lignes.append((
            w,
            pos_tags.get(w, ""),                 # POS
            int(compteur.get(w, 0)),            # fréquence pivot-centrée
            int(F12.get(w, 0)),                 # nb de fenêtres contenant pivot & w
            float(scores.get(w, 0.0))           # loglike
        ))
    df = pd.DataFrame(lignes, columns=["cooccurrent", "pos", "frequence", "fenetres_ensemble", "loglike"])
    df = df.sort_values(["loglike", "frequence"], ascending=[False, False]).reset_index(drop=True)

    with tab_resultats:
        st.markdown("### Résultats")
        st.caption("« fenetres_ensemble » = nombre d’unités de contexte contenant simultanément le pivot et le cooccurrent.")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="Télécharger le CSV",
            data=generer_csv(df).getvalue(),
            file_name="cooccurrences.csv",
            mime="text/csv"
        )

        if generer_nuage:
            st.markdown("### Nuage de mots (pondéré par la fréquence)")
            df_top = df[df["frequence"] > 0].sort_values(["frequence", "loglike"], ascending=[False, False]).head(int(top_n))
            freq_dict = dict(zip(df_top["cooccurrent"], df_top["frequence"]))
            generer_wordcloud(freq_dict, f"Top {int(top_n)} cooccurrences")
