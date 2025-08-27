# python -m streamlit run main.py
# pip install streamlit pandas matplotlib worldcloud scipy
# python -m spacy download fr_core_news_md

# -*- coding: utf-8 -*-
# Application Streamlit : Cooccurrences — Fréquences et Log-likelihood séparés + Concordancier
# Modes de fenêtre : Mots (±k), Phrase, Paragraphe (mode Document supprimé)
# Stopwords : option spaCy uniquement
# Nettoyage : exclure nombres, exclure mots d’1 lettre
# Règle apostrophe (détection) : « c'est » -> « est », « l'homme » -> « homme », gère aussi c ' est
# Pivot jamais filtré
# POS dans les tableaux
# Log-likelihood via SciPy (G²) sur les MÊMES fenêtres que les fréquences
# Concordancier : phrases complètes en surface (avec stopwords visibles), surlignage pivot + cooccurrent
# Deux blocs distincts d’affichage :
#   1) Fréquences (table + wordcloud + concordancier)
#   2) Log-likelihood (table + wordcloud + concordancier)
# Analyses figées via st.session_state pour éviter tout recalcul inutile

# ================================
# IMPORTS
# ================================
import io
import re
import html
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from wordcloud import WordCloud
import spacy
from scipy.stats import power_divergence

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
def construire_stopwords(appliquer_stop: bool):
    """Renvoie l’ensemble des stopwords (spaCy) si coché, sinon ensemble vide."""
    return set(nlp.Defaults.stop_words) if appliquer_stop else set()

# ================================
# SEGMENTATION PARAGRAPHES
# ================================
def segmenter_paragraphes(texte: str):
    """Paragraphe = bloc séparé par au moins une ligne vide."""
    norm = texte.replace("\r\n", "\n").replace("\r", "\n")
    blocs = re.split(r"\n\s*\n+", norm)
    return [b.strip() for b in blocs if b.strip()]

# ================================
# NORMALISATION (RÈGLE APOSTROPHE)
# ================================
APOS = {"'", "’"}

def normaliser_avec_apostrophe_joint(token_text: str) -> str:
    """Si le token contient une apostrophe, garder la partie à droite (ex. « l'homme » -> « homme »)."""
    if "'" in token_text or "’" in token_text:
        parts = re.split(r"[’']", token_text, maxsplit=1)
        if len(parts) == 2:
            return parts[1]
    return token_text

def iter_tokens_norm_et_carte(doc, stopset, pivot, exclure_nombres, exclure_monolettre):
    """
    Produit :
      - norm_list : formes normalisées utilisées pour l’analyse
      - spans_list : paires (start,end_excl) d’indices de tokens surface correspondant à chaque forme normalisée
    Sert pour : comptages, POS cohérents, et reconstruction surface du concordancier.
    """
    norm_list, spans_list = [], []
    toks = list(doc)
    i, n = 0, len(toks)
    while i < n:
        tok = toks[i]
        raw = tok.text
        low = raw.lower()

        # Mot + apostrophe + mot : on garde le mot de droite (couvre 3 tokens surface)
        if low.isalpha() and i + 2 < n and toks[i+1].text in APOS:
            droite = toks[i+2].text.lower()
            mot = droite
            if mot == pivot:
                if mot.isalnum():
                    norm_list.append(mot)
                    spans_list.append((i, i+3))
            else:
                if mot.isalnum() and not (exclure_nombres and mot.isdigit()) and not (exclure_monolettre and len(mot) == 1):
                    if mot not in stopset:
                        norm_list.append(mot)
                        spans_list.append((i, i+3))
            i += 3
            continue

        # Apostrophe dans le token
        mot = normaliser_avec_apostrophe_joint(raw).lower() if (("'" in raw) or ("’" in raw)) else low

        if mot == pivot:
            if mot.isalnum():
                norm_list.append(mot)
                spans_list.append((i, i+1))
        else:
            if mot.isalnum() and not (exclure_nombres and mot.isdigit()) and not (exclure_monolettre and len(mot) == 1):
                if mot not in stopset:
                    norm_list.append(mot)
                    spans_list.append((i, i+1))
        i += 1
    return norm_list, spans_list

def iter_tokens_normalises(doc, stopset, pivot, exclure_nombres, exclure_monolettre):
    """Version qui renvoie uniquement la liste normalisée (plus rapide pour compter)."""
    norm, _ = iter_tokens_norm_et_carte(doc, stopset, pivot, exclure_nombres, exclure_monolettre)
    return norm

# ================================
# FENÊTRES DE CONTEXTE
# ================================
def fenetres_mots(doc, pivot: str, k: int, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """Fenêtres ±k autour de chaque pivot (après normalisation/filtrage)."""
    seq = iter_tokens_normalises(doc, stopset, pivot, exclure_nombres, exclure_monolettre)
    indices = [i for i, w in enumerate(seq) if w == pivot]
    fenetres = []
    for idx in indices:
        d = max(0, idx - k)
        f = min(len(seq), idx + k + 1)
        fen = set(seq[d:f])
        if fen:
            fenetres.append(fen)
    return fenetres, seq  # on renvoie aussi la séquence pour les stats corpus

def fenetres_phrases(doc, stopset, pivot, exclure_nombres: bool, exclure_monolettre: bool):
    """Fenêtres = phrases spaCy ; normalisation/filtrage au niveau phrase, puis ensemble."""
    fenetres = []
    for sent in doc.sents:
        seq = iter_tokens_normalises(sent, stopset, pivot, exclure_nombres, exclure_monolettre)
        if seq:
            fenetres.append(set(seq))
    return fenetres

def fenetres_paragraphes(texte: str, stopset, pivot, exclure_nombres: bool, exclure_monolettre: bool):
    """Fenêtres = paragraphes séparés par ligne vide."""
    fenetres = []
    for pa in segmenter_paragraphes(texte):
        d = nlp(pa)
        seq = iter_tokens_normalises(d, stopset, pivot, exclure_nombres, exclure_monolettre)
        if seq:
            fenetres.append(set(seq))
    return fenetres

# ================================
# LOG-LIKELIHOOD (SciPy)
# ================================
def loglike_scipy_par_mot(T: int, F1: int, f2: int, a: int) -> float:
    """
    G² via SciPy (test de rapport de vraisemblance) sur la table 2×2 :
        a = fenêtres pivot & mot
        b = F1 - a
        c = f2 - a
        d = T - a - b - c
    On calcule les attentes sous indépendance et on appelle power_divergence(..., lambda_="log-likelihood").
    """
    b = F1 - a
    c = f2 - a
    d = T - a - b - c
    if min(a, b, c, d) < 0:
        return 0.0
    obs = np.array([[a, b], [c, d]], dtype=float)
    total = obs.sum()
    if total <= 0:
        return 0.0
    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    exp = (row_sums @ col_sums) / total
    stat, _ = power_divergence(obs, f_exp=exp, lambda_="log-likelihood", axis=None)
    if not np.isfinite(stat):
        return 0.0
    return float(max(stat, 0.0))

def compter_loglike_sur_fenetres(fenetres, pivot: str):
    """
    Calcule : scores G², T, F1, F12, F2 à partir d’un jeu de fenêtres (mêmes que pour les fréquences).
    Retourne : scores, T, F1, F12 (dict), F2 (Counter)
    """
    T = len(fenetres)
    if T == 0:
        return {}, 0, 0, {}, Counter()
    F1 = 0
    F2 = Counter()
    F12 = Counter()
    for S in fenetres:
        cp = pivot in S
        if cp:
            F1 += 1
        for w in S:
            F2[w] += 1
            if cp and w != pivot:
                F12[w] += 1
    scores = {}
    for w, f2 in F2.items():
        if w == pivot:
            continue
        a = F12[w]
        scores[w] = loglike_scipy_par_mot(T, F1, f2, a)
    return scores, T, F1, dict(F12), F2

# ================================
# UTILITAIRES (POS, CSV, WordCloud)
# ================================
def etiqueter_pos_corpus(textes, stopset, pivot, exclure_nombres: bool, exclure_monolettre: bool):
    """Donne le POS majoritaire par forme normalisée (cohérent avec l’analyse)."""
    pos_counts = defaultdict(Counter)
    for txt in textes:
        d = nlp(txt)
        norm_list, spans_list = iter_tokens_norm_et_carte(d, stopset, pivot, exclure_nombres, exclure_monolettre)
        toks = list(d)
        for norm, (s_i, _e_i) in zip(norm_list, spans_list):
            pos_counts[norm][toks[s_i].pos_] += 1
    return {w: ctr.most_common(1)[0][0] for w, ctr in pos_counts.items()}

def generer_csv(df):
    """Flux CSV téléchargeable."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def generer_wordcloud(freq_dict, titre):
    """Nuage de mots robuste (ignore valeurs nulles ou négatives)."""
    freq_pos = {w: float(v) for w, v in freq_dict.items() if v and v > 0}
    if not freq_pos:
        st.info("Nuage de mots non généré : aucune valeur strictement positive.")
        return
    # Normalisation WordCloud accepte des floats
    wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq_pos)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(titre)
    st.pyplot(fig)

# ================================
# CONCORDANCIER (phrases complètes en surface)
# ================================
def phrase_surface_html(sent, pivot, cible, stopset, exclure_nombres, exclure_monolettre):
    """
    Surligne PIVOT et CIBLE dans la phrase de surface.
    On derive les indices surface à partir de la carte normalisée -> surface, ce qui gère correctement les apostrophes.
    """
    toks = list(sent)
    norm_list, spans_list = iter_tokens_norm_et_carte(sent, stopset, pivot, exclure_nombres, exclure_monolettre)

    piv_idx = set()
    cib_idx = set()
    for norm, (s_i, e_i) in zip(norm_list, spans_list):
        head = max(e_i - 1, s_i)  # tête du span
        if norm == pivot:
            piv_idx.add(head)
        if cible and norm == cible:
            cib_idx.add(head)

    css = (
        "<style>"
        ".pivot-badge{background:#111;color:#fff;border-radius:4px;padding:0 4px}"
        ".cible-badge{background:#0b7285;color:#fff;border-radius:3px;padding:0 2px}"
        ".kwic-sent{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Noto Sans','Helvetica Neue',Arial;"
        "line-height:1.6;margin:6px 0}"
        "</style>"
    )

    out = []
    for i, tok in enumerate(toks):
        text = html.escape(tok.text_with_ws)
        if i in piv_idx:
            out.append(f"<span class='pivot-badge'>{text}</span>")
        elif i in cib_idx:
            out.append(f"<span class='cible-badge'>{text}</span>")
        else:
            out.append(text)

    return css + f"<div class='kwic-sent'>{''.join(out).strip()}</div>"

# ================================
# INTERFACE — TITRE + EXPLICATIONS
# ================================
st.set_page_config(page_title="Cooccurrences — Fréquences & Log-likelihood", layout="centered")
st.markdown("# Cooccurrences autour d’un mot pivot : fréquences et log-likelihood")

# Explications complètes (sous le titre)
st.markdown(
    "Cette application sépare **clairement** deux niveaux d’analyse :\n\n"
    "**1) Fréquences brutes** : on compte, pour un mot pivot donné, les mots qui apparaissent dans la même unité de contexte "
    "(appelée *fenêtre*) et combien de fois ils co-apparaissent. Les fenêtres peuvent être au choix :\n"
    "• **Mots (±k)** : pour chaque occurrence du pivot, on prend k mots avant et k mots après, après normalisation/filtrage.\n"
    "• **Phrase** : chaque phrase du texte (détectée par spaCy) forme une fenêtre.\n"
    "• **Paragraphe** : chaque paragraphe (blocs séparés par au moins une ligne vide) forme une fenêtre.\n\n"
    "Le pivot **n’est jamais filtré**. Les stopwords spaCy peuvent être retirés (optionnelle). "
    "On peut aussi retirer les nombres et les mots d’une lettre. Les formes avec apostrophe sont **détectées** de manière à garder "
    "uniquement la partie droite (ex. « l'homme » → « homme », « c'est » → « est »), ce qui évite que des lettres isolées "
    "se glissent dans les cooccurrents.\n\n"
    "**2) Test log-likelihood (G²)** : sur les **mêmes fenêtres** que les fréquences, on calcule un score statistique qui compare "
    "la cooccurrence observée à ce qu’on attendrait si le pivot et le mot étaient indépendants. Un score élevé indique une "
    "association **non due au hasard**. Attention : un score élevé peut signaler une **sur-association** (ensemble avec le pivot) "
    "ou une **sous-association** (le mot apparaît souvent ailleurs mais pas avec le pivot). Le score indique l’écart à l’indépendance, "
    "pas le sens de la corrélation.\n\n"
    "Les résultats sont présentés dans **deux tableaux distincts** : d’abord les **fréquences**, puis les **scores log-likelihood**. "
    "Chaque tableau est accompagné de son nuage de mots et de son concordancier (phrases complètes), afin de permettre une lecture "
    "à la fois quantitative et qualitative."
)

# ================================
# PARAMÈTRES D’ENTRÉE
# ================================
uploaded = st.file_uploader("Fichier texte (.txt)", type=["txt"])
texte_libre = st.text_area("Ou collez votre texte ici", height=220)

st.markdown("### Paramètres d’analyse")
pivot = st.text_input("Mot pivot (obligatoire)", value="soleil").strip().lower()
fenetre = st.selectbox("Fenêtre de contexte", ["Mots (±k)", "Phrase", "Paragraphe"])
k = 5
if fenetre == "Mots (±k)":
    k = st.number_input("Taille de la fenêtre k (mots de contexte)", min_value=1, max_value=100, value=5, step=1)

appliquer_stop = st.checkbox("Appliquer les stopwords (spaCy)", value=True)
exclure_nombres = st.checkbox("Exclure les nombres", value=True)
exclure_monolettre = st.checkbox("Exclure les mots d’une seule lettre", value=True)

# Gestion état
if "run_id" not in st.session_state:
    st.session_state["run_id"] = 0
if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False

# ================================
# ANALYSE (au clic)
# ================================
if st.button("Lancer l’analyse"):
    if not pivot:
        st.error("Veuillez saisir un mot pivot.")
        st.stop()

    texte = uploaded.read().decode("utf-8", errors="ignore") if uploaded else texte_libre
    if not texte or not texte.strip():
        st.error("Veuillez fournir un texte.")
        st.stop()

    stopset = construire_stopwords(appliquer_stop)
    doc = nlp(texte)

    # Fenêtres et taille corpus normalisée (pour information)
    if fenetre == "Mots (±k)":
        fenetres, seq_doc = fenetres_mots(doc, pivot, k, stopset, exclure_nombres, exclure_monolettre)
        total_mots_norm = len(seq_doc)
    elif fenetre == "Phrase":
        fenetres = fenetres_phrases(doc, stopset, pivot, exclure_nombres, exclure_monolettre)
        total_mots_norm = sum(len(iter_tokens_normalises(s, stopset, pivot, exclure_nombres, exclure_monolettre)) for s in doc.sents)
    else:  # Paragraphe
        fenetres = fenetres_paragraphes(texte, stopset, pivot, exclure_nombres, exclure_monolettre)
        total_mots_norm = 0
        for pa in segmenter_paragraphes(texte):
            dpa = nlp(pa)
            total_mots_norm += len(iter_tokens_normalises(dpa, stopset, pivot, exclure_nombres, exclure_monolettre))

    # Fréquences pivot-centrées (1 incrément par fenêtre où w co-apparaît avec le pivot)
    freq_counter = Counter()
    for S in fenetres:
        if pivot in S:
            for w in S:
                if w != pivot:
                    freq_counter[w] += 1

    # Log-likelihood sur les mêmes fenêtres
    scores, T, F1, F12, F2 = compter_loglike_sur_fenetres(fenetres, pivot)

    # POS majoritaire
    pos_tags = etiqueter_pos_corpus([texte], stopset, pivot, exclure_nombres, exclure_monolettre)

    # TABLEAU 1 — Fréquences
    coocs_freq = sorted(freq_counter.keys())
    lignes_freq = []
    for w in coocs_freq:
        lignes_freq.append((
            w,
            pos_tags.get(w, ""),
            int(freq_counter.get(w, 0)),
            int(F12.get(w, 0)),  # nb de fenêtres contenant pivot & w
        ))
    df_freq = pd.DataFrame(lignes_freq, columns=["cooccurrent", "pos", "frequence", "fenetres_ensemble"])
    df_freq = df_freq.sort_values(["frequence", "fenetres_ensemble"], ascending=[False, False]).reset_index(drop=True)

    # TABLEAU 2 — Log-likelihood (sur tous les mots vus dans les fenêtres, même si fréquence pivot=0)
    coocs_ll = sorted(w for w in scores.keys() if w != pivot)
    lignes_ll = []
    for w in coocs_ll:
        lignes_ll.append((
            w,
            pos_tags.get(w, ""),
            float(scores.get(w, 0.0)),
            int(F12.get(w, 0)),   # utile pour interprétation
        ))
    df_ll = pd.DataFrame(lignes_ll, columns=["cooccurrent", "pos", "loglike", "fenetres_ensemble"])
    df_ll = df_ll.sort_values(["loglike", "fenetres_ensemble"], ascending=[False, False]).reset_index(drop=True)

    # Concordancier : phrases en surface + listes normalisées
    sent_spans = list(doc.sents)
    sent_norms = [iter_tokens_normalises(s, stopset, pivot, exclure_nombres, exclure_monolettre) for s in sent_spans]
    contains_pivot = [pivot in set(nlist) for nlist in sent_norms]

    # Scores corpus/analyse
    nb_phrases = len(sent_spans)
    nb_paragraphes = len(segmenter_paragraphes(texte))
    nb_fenetres = T
    nb_fenetres_avec_pivot = F1
    nb_coocs_uniques_freq = len(df_freq)
    total_coocs_freq = int(df_freq["frequence"].sum())
    nb_coocs_uniques_ll = len(df_ll)

    # Sauvegarde en session
    st.session_state["run_id"] += 1
    st.session_state["analysis_ready"] = True
    st.session_state["pivot"] = pivot
    st.session_state["df_freq"] = df_freq
    st.session_state["df_ll"] = df_ll
    st.session_state["sent_spans"] = sent_spans
    st.session_state["sent_norms"] = sent_norms
    st.session_state["contains_pivot"] = contains_pivot
    st.session_state["stopset"] = stopset
    st.session_state["excl_num"] = exclure_nombres
    st.session_state["excl_1"] = exclure_monolettre
    st.session_state["texte"] = texte
    st.session_state["stats"] = {
        "total_mots_norm": int(total_mots_norm),
        "nb_phrases": int(nb_phrases),
        "nb_paragraphes": int(nb_paragraphes),
        "nb_fenetres": int(nb_fenetres),
        "nb_fenetres_avec_pivot": int(nb_fenetres_avec_pivot),
        "nb_coocs_uniques_freq": int(nb_coocs_uniques_freq),
        "total_coocs_freq": int(total_coocs_freq),
        "nb_coocs_uniques_ll": int(nb_coocs_uniques_ll),
    }

# ================================
# AFFICHAGE RÉSULTATS (sans recalcul)
# ================================
if st.session_state.get("analysis_ready", False):

    # Bloc de synthèse
    st.markdown("## Statistiques de l’analyse")
    s = st.session_state["stats"]
    st.write(
        f"Nombre de mots normalisés conservés : {s['total_mots_norm']}\n\n"
        f"Nombre de phrases : {s['nb_phrases']}\n\n"
        f"Nombre de paragraphes : {s['nb_paragraphes']}\n\n"
        f"Nombre total de fenêtres : {s['nb_fenetres']}\n\n"
        f"Fenêtres contenant le pivot (F1) : {s['nb_fenetres_avec_pivot']}\n\n"
        f"Cooccurrents uniques (table Fréquences) : {s['nb_coocs_uniques_freq']}\n\n"
        f"Total des cooccurrences (somme des fréquences) : {s['total_coocs_freq']}\n\n"
        f"Cooccurrents uniques (table Log-likelihood) : {s['nb_coocs_uniques_ll']}"
    )

    # ================
    # 1) FRÉQUENCES
    # ================
    st.markdown("## Tableau 1 — Fréquences pivot-centrées")
    st.caption("fréquence = nombre de fenêtres où le pivot et le mot co-apparaissent ; "
               "fenetres_ensemble = nombre de fenêtres contenant simultanément le pivot et le mot.")
    df_freq = st.session_state["df_freq"]
    st.dataframe(df_freq, use_container_width=True)

    st.download_button(
        label="Télécharger le CSV (Fréquences)",
        data=generer_csv(df_freq).getvalue(),
        file_name="cooccurrences_frequences.csv",
        mime="text/csv",
        key=f"dl_csv_freq_{st.session_state['run_id']}"
    )

    # Nuage de mots (pondéré par la fréquence)
    st.markdown("### Nuage de mots — pondéré par la fréquence")
    top_n_freq = st.number_input("Top N (fréquences)", min_value=1, max_value=500, value=10, step=1, key=f"top_wc_freq_{st.session_state['run_id']}")
    df_top_freq = df_freq[df_freq["frequence"] > 0].sort_values(["frequence", "fenetres_ensemble"], ascending=[False, False]).head(int(top_n_freq))
    freq_dict = dict(zip(df_top_freq["cooccurrent"], df_top_freq["frequence"]))
    generer_wordcloud(freq_dict, f"Top {int(top_n_freq)} cooccurrences (fréquence)")

    # Concordancier pour Fréquences
    st.markdown("### Concordancier (phrases) — sélection à partir du tableau Fréquences")
    coocs_list_freq = list(df_freq["cooccurrent"])
    if coocs_list_freq:
        cible_freq = st.selectbox("Choisir un cooccurrent (Fréquences)", coocs_list_freq, index=0, key=f"cc_freq_{st.session_state['run_id']}")
        nb_max_freq = st.number_input("Nombre maximum de phrases", min_value=1, max_value=5000, value=200, step=10, key=f"nbmax_freq_{st.session_state['run_id']}")
        pivot_cc = st.session_state["pivot"]
        sent_spans = st.session_state["sent_spans"]
        sent_norms = st.session_state["sent_norms"]
        stopset_cc = st.session_state["stopset"]
        excl_num = st.session_state["excl_num"]
        excl_1 = st.session_state["excl_1"]

        # Filtrage des phrases : pivot + cible présents (dans la forme normalisée)
        lignes_html = []
        n_affichees = 0
        for sent, nlist in zip(sent_spans, sent_norms):
            sset = set(nlist)
            if pivot_cc in sset and cible_freq in sset:
                lignes_html.append(phrase_surface_html(sent, pivot_cc, cible_freq, stopset_cc, excl_num, excl_1))
                n_affichees += 1
                if n_affichees >= int(nb_max_freq):
                    break

        if not lignes_html:
            st.info("Aucune phrase trouvée pour ce cooccurrent.")
        else:
            st.markdown("\n".join(lignes_html), unsafe_allow_html=True)
            bloc = "\n".join(lignes_html)
            doc_html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Concordancier (Fréquences)</title>"
                "</head><body>"
                f"<h3>Concordancier — pivot = {html.escape(pivot_cc)}, cooccurrent = {html.escape(cible_freq)}</h3>"
                f"{bloc}</body></html>"
            )
            st.download_button(
                label="Télécharger le concordancier (Fréquences, HTML)",
                data=doc_html.encode("utf-8"),
                file_name=f"concordancier_frequences_{pivot_cc}_{cible_freq}.html",
                mime="text/html",
                key=f"dl_kwic_freq_{st.session_state['run_id']}"
            )

    # =================
    # 2) LOG-LIKELIHOOD
    # =================
    st.markdown("## Tableau 2 — Scores log-likelihood (G²)")
    st.caption("Le score G² mesure l’écart à l’indépendance entre le pivot et le mot, sur les mêmes fenêtres que le tableau des fréquences. "
               "Un score élevé signale une association non due au hasard (sur- ou sous-association).")
    df_ll = st.session_state["df_ll"]
    st.dataframe(df_ll, use_container_width=True)

    st.download_button(
        label="Télécharger le CSV (Log-likelihood)",
        data=generer_csv(df_ll).getvalue(),
        file_name="cooccurrences_loglike.csv",
        mime="text/csv",
        key=f"dl_csv_ll_{st.session_state['run_id']}"
    )

    # Nuage de mots (pondéré par le score loglike)
    st.markdown("### Nuage de mots — pondéré par le score log-likelihood")
    top_n_ll = st.number_input("Top N (log-likelihood)", min_value=1, max_value=500, value=10, step=1, key=f"top_wc_ll_{st.session_state['run_id']}")
    df_top_ll = df_ll[df_ll["loglike"] > 0].sort_values(["loglike", "fenetres_ensemble"], ascending=[False, False]).head(int(top_n_ll))
    ll_dict = dict(zip(df_top_ll["cooccurrent"], df_top_ll["loglike"]))
    generer_wordcloud(ll_dict, f"Top {int(top_n_ll)} cooccurrences (log-likelihood)")

    # Concordancier pour Log-likelihood
    st.markdown("### Concordancier (phrases) — sélection à partir du tableau Log-likelihood")
    coocs_list_ll = list(df_ll["cooccurrent"])
    if coocs_list_ll:
        cible_ll = st.selectbox("Choisir un cooccurrent (Log-likelihood)", coocs_list_ll, index=0, key=f"cc_ll_{st.session_state['run_id']}")
        nb_max_ll = st.number_input("Nombre maximum de phrases", min_value=1, max_value=5000, value=200, step=10, key=f"nbmax_ll_{st.session_state['run_id']}")
        pivot_cc = st.session_state["pivot"]
        sent_spans = st.session_state["sent_spans"]
        sent_norms = st.session_state["sent_norms"]
        stopset_cc = st.session_state["stopset"]
        excl_num = st.session_state["excl_num"]
        excl_1 = st.session_state["excl_1"]

        lignes_html = []
        n_affichees = 0
        for sent, nlist in zip(sent_spans, sent_norms):
            sset = set(nlist)
            if pivot_cc in sset and cible_ll in sset:
                lignes_html.append(phrase_surface_html(sent, pivot_cc, cible_ll, stopset_cc, excl_num, excl_1))
                n_affichees += 1
                if n_affichees >= int(nb_max_ll):
                    break

        if not lignes_html:
            st.info("Aucune phrase trouvée pour ce cooccurrent.")
        else:
            st.markdown("\n".join(lignes_html), unsafe_allow_html=True)
            bloc = "\n".join(lignes_html)
            doc_html = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                "<title>Concordancier (Log-likelihood)</title>"
                "</head><body>"
                f"<h3>Concordancier — pivot = {html.escape(pivot_cc)}, cooccurrent = {html.escape(cible_ll)}</h3>"
                f"{bloc}</body></html>"
            )
            st.download_button(
                label="Télécharger le concordancier (Log-likelihood, HTML)",
                data=doc_html.encode("utf-8"),
                file_name=f"concordancier_loglike_{pivot_cc}_{cible_ll}.html",
                mime="text/html",
                key=f"dl_kwic_ll_{st.session_state['run_id']}"
            )

else:
    st.info("Lancez l’analyse pour afficher les tableaux, les nuages de mots et les concordanciers.")
