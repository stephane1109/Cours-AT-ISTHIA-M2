################################################
# Stéphane Meurisse
# www.codeandcortex.fr
# 28 aout 2025
################################################

# python -m streamlit run main.py

# ########## Dépendances
# pip install streamlit pandas matplotlib worldcloud pyvis scipy
# python -m spacy download fr_core_news_md
############

# Application Streamlit : Analyse des Cooccurrences par :
#     - 1. Fréquences
#     - 2. log-likelihood
#     - 3. lexico-syntaxiques (spaCy)

# (1)

# (2) Log-likelihood : mesurer la force des cooccurrences
# Le log-likelihood est une mesure statistique qui sert à tester l’indépendance entre deux mots.
# L’idée est de distinguer deux situations :
#       - les cooccurrences qui apparaissent par simple hasard, parce que les mots sont fréquents dans le corpus ;
#       - celles qui apparaissent beaucoup plus souvent que prévu, et qui révèlent donc une association significative.
# Le log-likelihood est donc une mesure qui permet de faire le tri :
# il indique si une cooccurrence est juste due à la fréquence des mots, ou si elle est anormalement fréquente et donc révélatrice d’un lien fort.
# En pratique, plus le score est élevé, plus l’association entre les deux mots est intéressante à interpréter.

# (3) L’analyse Lexico-syntaxique s’appuie sur les unités lexicales liées par des relations grammaticales et met en avant les dépendances

# L'application utilise le modèle medium de SPACY pour traiter les stopsword, les Pos-tag, et l'approche lexico-syntaxique

# - Affichage des résultats : 1) Fréquences, 2) Log-likelihood, 3) Analyse lexico-syntaxique.

# - Paramètres sous le texte, pas de barre latérale.
# - Résultats en dessous (onglets Résultats, Lexique, Explications).
# - 3 modes de fenêtres au choix : Mots (±k), Phrase (ponctuation), Paragraphe (retour à la ligne).
# - Stopwords : option spaCy (sans ajout manuel). Nettoyage optionnel des nombres et des mots d’1 lettre.
# - Apostrophes : « c’est » -> « est », « l’homme » -> « homme »
# - le mot pivot n’est jamais filtré.
# - POS affichées dans les tableaux de cooccurrents. Log-likelihood via SciPy sur les mêmes fenêtres que les fréquences.
# - Concordanciers : Fréquences et Log-likelihood (phrases surface), Syntaxique aligné (relations du pivot uniquement).
# - Tout est téléchargeables en HTML autonome.
# - Graphes PyVis : Fréquence (label = fréquence), Loglike (label = G²), Syntaxique (label = relation spaCy).

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
from pyvis.network import Network
from streamlit.components.v1 import html as st_html
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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
    """Renvoie l’ensemble des stopwords spaCy si coché, sinon ensemble vide."""
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
    Produit deux listes alignées :
      - norm_list : formes normalisées utilisées pour l’analyse,
      - spans_list : paires (start, end_excl) d’indices surface.
    Sert aux comptages et au repérage surface pour les concordanciers.
    """
    norm_list, spans_list = [], []
    toks = list(doc)
    i, n = 0, len(toks)
    while i < n:
        tok = toks[i]
        raw = tok.text
        low = raw.lower()

        # Cas "c ' est" -> garder "est"
        if low.isalpha() and i + 2 < n and toks[i+1].text in APOS:
            droite = toks[i+2].text.lower()
            mot = droite
            if mot == pivot:
                if mot.isalnum():
                    norm_list.append(mot); spans_list.append((i, i+3))
            else:
                if mot.isalnum() and not (exclure_nombres and mot.isdigit()) and not (exclure_monolettre and len(mot) == 1):
                    if mot not in stopset:
                        norm_list.append(mot); spans_list.append((i, i+3))
            i += 3
            continue

        # Cas "l'homme" -> "homme"
        mot = normaliser_avec_apostrophe_joint(raw).lower() if (("'" in raw) or ("’" in raw)) else low

        if mot == pivot:
            if mot.isalnum():
                norm_list.append(mot); spans_list.append((i, i+1))
        else:
            if mot.isalnum() and not (exclure_nombres and mot.isdigit()) and not (exclure_monolettre and len(mot) == 1):
                if mot not in stopset:
                    norm_list.append(mot); spans_list.append((i, i+1))
        i += 1
    return norm_list, spans_list

def iter_tokens_normalises(doc, stopset, pivot, exclure_nombres, exclure_monolettre):
    """Renvoie uniquement la liste normalisée (plus rapide pour les comptages)."""
    norm, _ = iter_tokens_norm_et_carte(doc, stopset, pivot, exclure_nombres, exclure_monolettre)
    return norm

# ================================
# FENÊTRES LINÉAIRES (FRÉQUENCES & LOGLIKE)
# ================================
def fenetres_mots(doc, pivot: str, k: int, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """Fenêtres ±k autour de chaque pivot (après normalisation/filtrage)."""
    seq = iter_tokens_normalises(doc, stopset, pivot, exclure_nombres, exclure_monolettre)
    indices = [i for i, w in enumerate(seq) if w == pivot]
    fenetres = []
    for idx in indices:
        d = max(0, idx - k); f = min(len(seq), idx + k + 1)
        fen = set(seq[d:f])
        if fen:
            fenetres.append(fen)
    return fenetres, seq

def fenetres_phrases(doc, stopset, pivot, exclure_nombres: bool, exclure_monolettre: bool):
    """Fenêtres = phrases spaCy (après normalisation/filtrage par phrase)."""
    fenetres = []
    for sent in doc.sents:
        seq = iter_tokens_normalises(sent, stopset, pivot, exclure_nombres, exclure_monolettre)
        if seq:
            fenetres.append(set(seq))
    return fenetres

def fenetres_paragraphes(texte: str, stopset, pivot, exclure_nombres: bool, exclure_monolettre: bool):
    """Fenêtres = paragraphes séparés par ligne vide (après normalisation/filtrage)."""
    fenetres = []
    for pa in segmenter_paragraphes(texte):
        d = nlp(pa)
        seq = iter_tokens_normalises(d, stopset, pivot, exclure_nombres, exclure_monolettre)
        if seq:
            fenetres.append(set(seq))
    return fenetres

# ================================
# COOCCURRENCES LEXICO-SYNTAXIQUES
# ================================
def extraire_cooc_syntaxiques_doc(doc, pivot: str, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """
    Pour chaque pivot surface : enfants (child.dep_) + tête (tok.dep_ vers head).
    Retour :
      - compteur_pairs : Counter{ (mot_norm, relation_spacy) -> fréquence }
      - index_phrase_pairs : liste par phrase des paires (mot_norm, relation_spacy) pour filtrage du concordancier
    """
    compteur_pairs = Counter()
    index_phrase_pairs = []

    for sent in doc.sents:
        pairs = []
        for tok in sent:
            tok_norm = normaliser_avec_apostrophe_joint(tok.text).lower()
            if tok_norm != pivot:
                continue
            # Enfants du pivot
            for child in tok.children:
                w = normaliser_avec_apostrophe_joint(child.text).lower()
                if not w.isalnum():
                    continue
                if (exclure_nombres and w.isdigit()) or (exclure_monolettre and len(w) == 1):
                    continue
                if w in stopset or w == pivot:
                    continue
                rel = child.dep_
                pairs.append((w, rel)); compteur_pairs[(w, rel)] += 1
            # Tête du pivot (si pas racine)
            if tok.head is not None and tok.head != tok:
                w = normaliser_avec_apostrophe_joint(tok.head.text).lower()
                if w.isalnum() and w != pivot and not (exclure_nombres and w.isdigit()) and not (exclure_monolettre and len(w) == 1) and w not in stopset:
                    rel = tok.dep_
                    pairs.append((w, rel)); compteur_pairs[(w, rel)] += 1
        index_phrase_pairs.append(pairs)

    return compteur_pairs, index_phrase_pairs

# ================================
# LOG-LIKELIHOOD (avec la librairie SciPy)
# ================================
def loglike_scipy_par_mot(T: int, F1: int, f2: int, a: int) -> float:
    """G² via SciPy (rapport de vraisemblance) sur la table 2×2."""
    b = F1 - a; c = f2 - a; d = T - a - b - c
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
    """Calcule scores G², T, F1, F12, F2 à partir d’un jeu de fenêtres."""
    T = len(fenetres)
    if T == 0:
        return {}, 0, 0, {}, Counter()
    F1 = 0; F2 = Counter(); F12 = Counter()
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
# UTILITAIRES (POS, CSV, WordCloud, PyVis, Explications, HTML)
# ================================
def etiqueter_pos_corpus(textes, stopset, pivot, exclure_nombres: bool, exclure_monolettre: bool):
    """POS majoritaire par forme normalisée (cohérent avec l’analyse)."""
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
    """Nuage de mots robuste (ignore valeurs ≤ 0)."""
    freq_pos = {w: float(v) for w, v in freq_dict.items() if v and v > 0}
    if not freq_pos:
        st.info("Nuage de mots non généré : aucune valeur strictement positive.")
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq_pos)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); plt.title(titre)
    st.pyplot(fig)

def pyvis_reseau_html(pivot: str, poids_dict, titre: str, top_n: int = 30, syntaxique: bool = False, mode_label: str = "freq", edge_label_size: int = 10):
    """Construit un réseau centré sur le pivot et renvoie l'HTML PyVis."""
    net = Network(height="600px", width="100%", directed=False, notebook=False)
    net.barnes_hut()
    net.set_options("""{
      "edges": {"color": {"inherit": false}, "smooth": false},
      "nodes": {"shape": "dot", "scaling": {"min": 10, "max": 50}},
      "physics": {"stabilization": true}
    }""")

    net.add_node(pivot, label=pivot, color="#e63946", title=pivot, value=50)

    cmap = cm.get_cmap("tab10")
    couleurs = [mcolors.rgb2hex(cmap(i % 10)) for i in range(100)]

    items = sorted(poids_dict.items(), key=lambda x: x[1], reverse=True)[:max(1, int(top_n))]
    if not items:
        return "<div>Aucune arête.</div>"

    vals = [float(p) for (_k, p) in items]
    wmin, wmax = min(vals), max(vals)

    def width_scale(v):
        if wmax == wmin:
            return 2
        return 1 + 5 * ((v - wmin) / (wmax - wmin))

    if syntaxique:
        for idx, ((w, rel), p) in enumerate(items):
            couleur = couleurs[idx % len(couleurs)]
            net.add_node(w, label=w, title=w, value=20, color=couleur)
            label = rel if rel else ""
            net.add_edge(
                pivot, w,
                value=float(p),
                width=width_scale(float(p)),
                label=label,
                title=f"{label} — {p}",
                font={"size": edge_label_size}
            )
    else:
        for idx, (w, p) in enumerate(items):
            couleur = couleurs[idx % len(couleurs)]
            net.add_node(w, label=w, title=w, value=20, color=couleur)
            edge_label = f"{int(p)}" if mode_label == "freq" else f"{float(p):.1f}"
            net.add_edge(
                pivot, w,
                value=float(p),
                width=width_scale(float(p)),
                label=edge_label,
                title=edge_label,
                font={"size": edge_label_size}
            )

    return net.generate_html()

# ================================
# TABLES EXPLICATIVES ENRICHIES — POS et RELATIONS (spaCy/UD) avec définition FR + exemple
# ================================
def table_pos_explicative_fr_enrichie() -> pd.DataFrame:
    """POS spaCy/UD : label, définition synthétique (FR), exemple court."""
    data = [
        ("ADJ",   "adjectif modifiant un nom",                    "une hausse économique notable"),
        ("ADP",   "adposition (préposition, postposition)",       "hausse de la fréquentation"),
        ("ADV",   "adverbe modifiant verbe/adj./phrase",          "Paris attire fortement"),
        ("AUX",   "auxiliaire dans les périphrases verbales",     "a été annoncé"),
        ("CCONJ", "conjonction de coordination",                  "tourisme et économie"),
        ("SCONJ", "conjonction de subordination",                 "parce que la demande augmente"),
        ("DET",   "déterminant lié à un nom",                     "la fréquentation"),
        ("INTJ",  "interjection",                                 "oh, quelle affluence"),
        ("NOUN",  "nom commun",                                   "tourisme, fréquentation, hausse"),
        ("PROPN", "nom propre",                                   "Paris, Île-de-France"),
        ("NUM",   "numéral",                                       "plus de 10 millions"),
        ("PART",  "particule/particule grammaticale",             "ne ... pas"),
        ("PRON",  "pronom",                                        "ils augmentent"),
        ("PUNCT", "ponctuation",                                   "— , . ; :"),
        ("SYM",   "symbole",                                       "%, €"),
        ("VERB",  "verbe lexical",                                 "augmente, progresse"),
        ("X",     "autres/inconnu",                                "token hétéroclite"),
        ("SPACE", "espace (blanc)",                                "espacement")
    ]
    return pd.DataFrame(data, columns=["POS", "Définition (FR)", "Exemple"])

def table_dep_explicative_fr_enrichie() -> pd.DataFrame:
    """
    Relations de dépendance UD/spaCy : label, définition FR fidèle, exemple FR minimal.
    Les exemples sont schématiques et visent l’intuition.
    """
    data = [
        ("nsubj",      "sujet nominal d’un verbe/adj.",                          "Paris attire des visiteurs → Paris"),
        ("obj",        "objet direct du verbe",                                  "attire des visiteurs → visiteurs"),
        ("iobj",       "objet indirect (souvent datif)",                         "parle aux touristes → aux touristes"),
        ("obl",        "complément oblique/prépositionnel",                      "progresse en été → en été"),
        ("advmod",     "modifieur adverbial",                                    "augmente fortement → fortement"),
        ("amod",       "modifieur adjectival du nom",                            "hausse notable → notable"),
        ("nmod",       "modifieur nominal (nom de nom)",                         "fréquentation de musées → musées"),
        ("appos",      "apposition",                                             "Paris, capitale française → capitale"),
        ("det",        "déterminant du nom",                                     "la fréquentation → la"),
        ("case",       "marqueur casuel/préposition du syntagme nominal",        "de Paris → de"),
        ("compound",   "élément de mot composé",                                 "Île-de-France → Île-de"),
        ("fixed",      "locution figée (mots fixes)",                            "afin de, parce que"),
        ("flat",       "groupe ‘plat’ (noms propres, dates…)",                   "Charles de Gaulle"),
        ("conj",       "élément coordonné",                                      "tourisme et économie → économie"),
        ("cc",         "coordinant (et, ou, mais…)",                             "tourisme et économie → et"),
        ("mark",       "subordonnant introduisant une subordonnée",              "parce que la demande… → parce que"),
        ("aux",        "auxiliaire (temps, passif…)",                            "a été annoncé → a, été"),
        ("cop",        "copule (verbe être copule)",                             "Paris est attractif → est"),
        ("root",       "racine de la phrase",                                    "Noeud central de l’arbre"),
        ("ccomp",      "complétive à verbe conjugué (objet clausal)",            "il affirme que la hausse continue → que…continue"),
        ("xcomp",      "complément clausal sans sujet propre (souvent infinitif)", "veut attirer plus → attirer"),
        ("acl",        "proposition relative/complément verbal du nom",          "touristes visitant Paris → visitant"),
        ("advcl",      "subordonnée adverbiale (temps, cause, condition…)",      "quand la saison commence"),
        ("csubj",      "sujet clausal (subordonnée sujet)",                      "qu’augmente la demande surprend"),
        ("expl",       "explétif",                                               "il y a, il est arrivé que… → il"),
        ("parataxis",  "parataxe (propositions juxtaposées)",                    "les chiffres montent, bonne nouvelle"),
        ("orphan",     "orphelin (ellipse, constructions incomplètes)",          "réponses télégraphiques"),
        ("punct",      "ponctuation",                                            "virgules, points"),
        ("dep",        "dépendance non classée (divers)",                        "secours pour cas atypiques")
    ]
    return pd.DataFrame(data, columns=["Relation", "Définition (FR)", "Exemple"])


def table_dep_explicative():
    """Table des dépendances spaCy/UD les plus fréquentes (onglet Explications)."""
    data = [
        ("nsubj", "sujet nominal"), ("obj", "objet direct"), ("iobj", "objet indirect"),
        ("obl", "complément circonstanciel"), ("amod", "modifieur adjectival"),
        ("advmod", "modifieur adverbial"), ("nmod", "modifieur nominal"),
        ("appos", "apposition"), ("det", "déterminant"), ("case", "marqueur casuel"),
        ("compound", "mot composé"), ("fixed", "expression figée"), ("flat", "locution plate/nom propre"),
        ("conj", "coordonné"), ("cc", "coordonnant"), ("mark", "subordonnant"),
        ("aux", "auxiliaire"), ("cop", "copule"), ("root", "racine")
    ]
    return pd.DataFrame(data, columns=["Relation spaCy", "Description"])

# ================================
# CONCORDANCIERS HTML
# ================================
def phrase_surface_html(sent, pivot, cible, stopset, exclure_nombres, exclure_monolettre):
    """Concordancier « phrase surface » : surlignage PIVOT et CIBLE."""
    toks = list(sent)
    norm_list, spans_list = iter_tokens_norm_et_carte(sent, stopset, pivot, exclure_nombres, exclure_monolettre)
    piv_idx = set(); cib_idx = set()
    for norm, (s_i, e_i) in zip(norm_list, spans_list):
        head = max(e_i - 1, s_i)
        if norm == pivot:
            piv_idx.add(head)
        if cible and norm == cible:
            cib_idx.add(head)
    css = (
        "<style>"
        ".pivot-badge{background:#e63946;color:#fff;border-radius:4px;padding:0 4px}"
        ".cible-badge{background:#1d3557;color:#fff;border-radius:3px;padding:0 2px}"
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

def document_html_kwic(titre: str, lignes_html):
    """Document HTML autonome pour téléchargement d’un concordancier."""
    style = (
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Noto Sans','Helvetica Neue',Arial;line-height:1.6;padding:12px}"
        "h3{margin-top:0}"
        ".pivot-badge{background:#e63946;color:#fff;border-radius:4px;padding:0 4px}"
        ".cible-badge{background:#1d3557;color:#fff;border-radius:3px;padding:0 2px}"
        ".kwic-sent{margin:6px 0}"
        ".ligne-phrase{margin:8px 0 4px 0;white-space:nowrap;overflow-x:auto}"
        ".grille-tokens{display:inline-flex;gap:12px;align-items:flex-start}"
        ".token-bloc{display:inline-flex;flex-direction:column;align-items:center}"
        ".mot{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono','Courier New', monospace;font-size:15px;padding:2px 4px;border-bottom:1px solid #bbb;border-radius:3px}"
        ".etiquette{font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono','Courier New', monospace;font-size:12px;color:#444}"
        ".stop{opacity:0.45}"
        ".pivot{background:#e63946;color:#fff}"
        ".cible{background:#1d3557;color:#fff}"
        "</style>"
    )
    corps = "\n".join(lignes_html)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(titre)}</title>{style}</head><body>"
        f"<h3>{html.escape(titre)}</h3>"
        f"{corps}</body></html>"
    )

# ================================
# CONCORDANCIER SYNTAXIQUE « ALIGNÉ — RELATIONS DU PIVOT »
# ================================
def html_relations_pivot_aligne(sent, pivot: str, cible: str = "", afficher_stop: bool = True):
    """
    Affiche une phrase en « grille » : mot au-dessus, ETIQUETTE = relation spaCy sous le mot
    UNIQUEMENT si ce mot est relié directement au pivot (enfant du pivot : child.dep_, ou tête du pivot : pivot.dep_).
    Le pivot et la cible sont surlignés. Les stopwords peuvent être masqués visuellement.
    """
    css = """
    <style>
    .ligne-phrase { margin: 8px 0 4px 0; white-space: nowrap; overflow-x: auto; }
    .grille-tokens { display: inline-flex; gap: 12px; align-items: flex-start; }
    .token-bloc { display: inline-flex; flex-direction: column; align-items: center; }
    .mot { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
           font-size: 15px; padding: 2px 4px; border-bottom: 1px solid #bbb; border-radius: 3px; }
    .etiquette { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
                 font-size: 12px; color: #444; min-height: 1em; }
    .stop { opacity: 0.45; }
    .pivot { background: #e63946; color: #fff; }
    .cible { background: #1d3557; color: #fff; }
    </style>
    """
    pivots = [t for t in sent if normaliser_avec_apostrophe_joint(t.text).lower() == pivot]
    voisins = {}  # token -> étiquette dep à afficher
    for p in pivots:
        for child in p.children:
            lab = child.dep_
            if lab:
                voisins[child] = lab
        if p.head is not None and p.head != p and p.dep_:
            voisins[p.head] = p.dep_

    blocs = []
    for t in sent:
        if t.is_space or not t.is_alpha:
            continue
        if not afficher_stop and t.is_stop:
            continue

        cls_mark = ""
        tnorm = normaliser_avec_apostrophe_joint(t.text).lower()
        if tnorm == pivot:
            cls_mark = " pivot"
        elif cible and tnorm == cible:
            cls_mark = " cible"

        lab = voisins.get(t, "")
        cls_stop = " stop" if t.is_stop else ""

        blocs.append(
            f'<div class="token-bloc{cls_stop}">'
            f'<div class="mot{cls_mark}">{html.escape(t.text)}</div>'
            f'<div class="etiquette">{html.escape(lab)}</div>'
            f'</div>'
        )
    return css + f'<div class="ligne-phrase"><div class="grille-tokens">{"".join(blocs)}</div></div>'

# ================================
# INTERFACE — TITRE + EXPLICATIONS
# ================================
st.set_page_config(page_title="Cooccurrences — Fréquences, Syntaxiques & Log-likelihood", layout="centered")
st.markdown("# Cooccurrences autour d’un mot pivot : fréquences, log-likelihood, lexico-syntaxique")
st.markdown(
    "Cette application sépare les fréquences brutes, le score de log-likelihood et l’analyse lexico-syntaxique.  \n"
    "Les fenêtres pour les fréquences et le log-likelihood peuvent être définies en mots (±k), en phrase ou en paragraphe.  \n"
    "Le mot pivot n’est jamais filtré.  \n"
    "Les stopwords spaCy, les nombres et les mots d’une lettre peuvent être exclus.  \n"
    "Les formes avec apostrophe sont normalisées en conservant la partie à droite.  \n"
)

# ================================
# PARAMÈTRES D’ANALYSE
# ================================
uploaded = st.file_uploader("Fichier texte (.txt)", type=["txt"])
texte_libre = st.text_area("Ou collez votre texte ici", height=220)

st.markdown("## Paramètres d’analyse")
pivot = st.text_input("Mot pivot (obligatoire)", value="soleil").strip().lower()
fenetre = st.selectbox("Fenêtre de contexte pour Fréquences et Log-likelihood", ["Mots (±k)", "Phrase", "Paragraphe"])
k = 5
if fenetre == "Mots (±k)":
    k = st.number_input("Taille de la fenêtre k (mots de contexte)", min_value=1, max_value=100, value=5, step=1)

appliquer_stop = st.checkbox("Appliquer les stopwords (spaCy)", value=True)
exclure_nombres = st.checkbox("Exclure les nombres", value=True)
exclure_monolettre = st.checkbox("Exclure les mots d’une seule lettre", value=True)

# État
if "run_id" not in st.session_state:
    st.session_state["run_id"] = 0
if "analysis_ready" not in st.session_state:
    st.session_state["analysis_ready"] = False

# ================================
# ANALYSE
# ================================
if st.button("Lancer l’analyse"):
    if not pivot:
        st.error("Veuillez saisir un mot pivot."); st.stop()

    texte = uploaded.read().decode("utf-8", errors="ignore") if uploaded else texte_libre
    if not texte or not texte.strip():
        st.error("Veuillez fournir un texte."); st.stop()

    stopset = construire_stopwords(appliquer_stop)
    doc = nlp(texte)

    # Fenêtres linéaires
    if fenetre == "Mots (±k)":
        fenetres, seq_doc = fenetres_mots(doc, pivot, k, stopset, exclure_nombres, exclure_monolettre)
        total_mots_norm = len(seq_doc)
    elif fenetre == "Phrase":
        fenetres = fenetres_phrases(doc, stopset, pivot, exclure_nombres, exclure_monolettre)
        total_mots_norm = sum(len(iter_tokens_normalises(s, stopset, pivot, exclure_nombres, exclure_monolettre)) for s in doc.sents)
    else:
        fenetres = fenetres_paragraphes(texte, stopset, pivot, exclure_nombres, exclure_monolettre)
        total_mots_norm = 0
        for pa in segmenter_paragraphes(texte):
            dpa = nlp(pa)
            total_mots_norm += len(iter_tokens_normalises(dpa, stopset, pivot, exclure_nombres, exclure_monolettre))

    # Fréquences pivot-centrées (1 par fenêtre où le pivot et w co-apparaissent)
    freq_counter = Counter()
    for S in fenetres:
        if pivot in S:
            for w in S:
                if w != pivot:
                    freq_counter[w] += 1

    # Syntaxiques (relations directes au pivot)
    compteur_pairs, index_phrase_pairs = extraire_cooc_syntaxiques_doc(
        doc, pivot, stopset, exclure_nombres, exclure_monolettre
    )

    # Log-likelihood sur les mêmes fenêtres
    scores, T, F1, F12, F2 = compter_loglike_sur_fenetres(fenetres, pivot)

    # POS pour tableaux
    pos_tags = etiqueter_pos_corpus([texte], stopset, pivot, exclure_nombres, exclure_monolettre)

    # TABLEAU — Fréquences
    df_freq = pd.DataFrame(
        [(w, pos_tags.get(w, ""), int(freq_counter[w]), int(F12.get(w, 0)))
         for w in sorted(freq_counter.keys())],
        columns=["cooccurrent", "pos", "frequence", "fenetres_ensemble"]
    ).sort_values(["frequence", "fenetres_ensemble"], ascending=[False, False]).reset_index(drop=True)

    # TABLEAU — Log-likelihood
    df_ll = pd.DataFrame(
        [(w, pos_tags.get(w, ""), float(scores[w]), int(F12.get(w, 0)))
         for w in sorted([x for x in scores.keys() if x != pivot])],
        columns=["cooccurrent", "pos", "loglike", "fenetres_ensemble"]
    ).sort_values(["loglike", "fenetres_ensemble"], ascending=[False, False]).reset_index(drop=True)

    # TABLEAU — Syntaxiques (une ligne par (mot, relation_spacy))
    df_syn = pd.DataFrame(
        [(w, pos_tags.get(w, ""), rel, int(c)) for (w, rel), c in sorted(compteur_pairs.items(), key=lambda x: x[1], reverse=True)],
        columns=["cooccurrent", "pos", "relation_spacy", "frequence"]
    ).sort_values(["frequence"], ascending=[False]).reset_index(drop=True)

    # Concordanciers : listes de phrases
    sent_spans = list(doc.sents)

    # Statistiques globales
    nb_phrases = len(sent_spans)
    nb_paragraphes = len(segmenter_paragraphes(texte))
    nb_fenetres = T
    nb_fenetres_avec_pivot = F1
    nb_coocs_uniques_freq = len(df_freq)
    total_coocs_freq = int(df_freq["frequence"].sum())
    nb_coocs_uniques_ll = len(df_ll)
    nb_coocs_uniques_syn = len(df_syn)

    # Lexique (formes et relations rencontrées)
    pos_map = pos_tags
    df_lex_formes = pd.DataFrame(sorted([(w, pos_map.get(w, "")) for w in pos_map.keys()], key=lambda x: x[0]),
                                 columns=["forme_norm", "pos"])
    rel_counts = Counter(); exemples = defaultdict(Counter)
    for (w, rel), c in compteur_pairs.items():
        rel_counts[rel] += c; exemples[rel][w] += c
    df_lex_rel = pd.DataFrame(
        [(rel, int(total), ", ".join([w for w, _ in exemples[rel].most_common(3)]))
         for rel, total in rel_counts.most_common()],
        columns=["relation_spacy", "occurrences", "exemples"]
    )

    # Tables explicatives
    df_pos_exp = table_pos_explicative_fr_enrichie()
    df_dep_exp = table_dep_explicative_fr_enrichie()

    # État session
    st.session_state["run_id"] += 1
    st.session_state["analysis_ready"] = True
    st.session_state["pivot"] = pivot
    st.session_state["df_freq"] = df_freq
    st.session_state["df_ll"] = df_ll
    st.session_state["df_syn"] = df_syn
    st.session_state["sent_spans"] = sent_spans
    st.session_state["index_phrase_pairs"] = index_phrase_pairs
    st.session_state["stopset"] = stopset
    st.session_state["excl_num"] = exclure_nombres
    st.session_state["excl_1"] = exclure_monolettre
    st.session_state["stats"] = {
        "total_mots_norm": int(total_mots_norm),
        "nb_phrases": int(nb_phrases),
        "nb_paragraphes": int(nb_paragraphes),
        "nb_fenetres": int(nb_fenetres),
        "nb_fenetres_avec_pivot": int(nb_fenetres_avec_pivot),
        "nb_coocs_uniques_freq": int(nb_coocs_uniques_freq),
        "total_coocs_freq": int(total_coocs_freq),
        "nb_coocs_uniques_ll": int(nb_coocs_uniques_ll),
        "nb_coocs_uniques_syn": int(nb_coocs_uniques_syn),
    }
    st.session_state["df_lex_formes"] = df_lex_formes
    st.session_state["df_lex_rel"] = df_lex_rel
    st.session_state["df_pos_exp"] = df_pos_exp
    st.session_state["df_dep_exp"] = df_dep_exp
    st.session_state["df_pos_exp"] = table_pos_explicative_fr_enrichie()
    st.session_state["df_dep_exp"] = table_dep_explicative_fr_enrichie()

# ================================
# AFFICHAGE RÉSULTATS + ONGLETS
# ================================
if st.session_state.get("analysis_ready", False):

    ong_res, ong_lex, ong_exp = st.tabs(["Résultats", "Lexique", "Explications"])

    with ong_res:
        st.markdown("## Statistiques de l’analyse")
        s = st.session_state["stats"]
        st.write(
            f"Nombre de mots normalisés conservés : {s['total_mots_norm']}\n\n"
            f"Nombre de phrases : {s['nb_phrases']}\n\n"
            f"Nombre de paragraphes : {s['nb_paragraphes']}\n\n"
            f"Nombre total de fenêtres (linéaires) : {s['nb_fenetres']}\n\n"
            f"Fenêtres contenant le pivot (F1) : {s['nb_fenetres_avec_pivot']}\n\n"
            f"Cooccurrents uniques (Fréquences) : {s['nb_coocs_uniques_freq']}\n\n"
            f"Total des cooccurrences (Fréquences) : {s['total_coocs_freq']}\n\n"
            f"Cooccurrents uniques (Log-likelihood) : {s['nb_coocs_uniques_ll']}\n\n"
            f"Cooccurrents uniques (Syntaxiques) : {s['nb_coocs_uniques_syn']}"
        )

        pivot_cc = st.session_state["pivot"]
        sent_spans = st.session_state["sent_spans"]
        stopset_cc = st.session_state["stopset"]
        excl_num = st.session_state["excl_num"]
        excl_1 = st.session_state["excl_1"]

        # =====================================
        # 1) FRÉQUENCES (tableau, nuage, graphe, concordancier)
        # =====================================
        st.markdown("# 1 — Fréquences à partir du mot pivot")
        st.markdown(
            "Ici, on analyse simplement les **cooccurrences récurrentes**, c’est-à-dire les mots qui apparaissent souvent "
            "dans le même contexte que le pivot.\n\n"
            "Plus la fréquence est élevée, plus cela indique que ces mots sont régulièrement associés dans le texte.\n\n"
            "Cette approche ne dit pas si l’association est due au hasard ou non : elle permet surtout d’observer "
            "les répétitions les plus visibles dans un corpus."
        )
        st.caption("frequence = nombre de fenêtres où pivot et mot co-apparaissent ; fenetres_ensemble = nombre de fenêtres contenant simultanément pivot et mot.")
        df_freq = st.session_state["df_freq"]
        st.dataframe(df_freq, use_container_width=True)
        st.download_button(
            label="Télécharger le CSV (Fréquences)",
            data=generer_csv(df_freq).getvalue(),
            file_name="cooccurrences_frequences.csv",
            mime="text/csv",
            key=f"dl_csv_freq_{st.session_state['run_id']}"
        )

        st.markdown("### Nuage de mots — pondéré par la fréquence")
        top_n_freq = st.number_input("Top N (fréquences)", min_value=1, max_value=500, value=10, step=1, key=f"top_wc_freq_{st.session_state['run_id']}")
        df_top_freq = df_freq[df_freq["frequence"] > 0].sort_values(["frequence", "fenetres_ensemble"], ascending=[False, False]).head(int(top_n_freq))
        generer_wordcloud(dict(zip(df_top_freq["cooccurrent"], df_top_freq["frequence"])), f"Top {int(top_n_freq)} cooccurrences (fréquence)")

        st.markdown("### Graphe interactif — cooccurrences (fréquence)")
        n_edges_freq = st.number_input("Nombre d’arêtes (Top-N) – fréquence", min_value=1, max_value=200, value=30, step=1, key=f"nedges_freq_{st.session_state['run_id']}")
        poids_freq = dict(zip(df_freq["cooccurrent"], df_freq["frequence"]))
        html_freq = pyvis_reseau_html(pivot_cc, poids_freq, "Réseau — Fréquence", top_n=int(n_edges_freq), syntaxique=False, mode_label="freq", edge_label_size=11)
        st_html(html_freq, height=620, scrolling=True)

        st.markdown("### Concordancier — à partir des fréquences")
        coocs_list_freq = list(df_freq["cooccurrent"])
        if coocs_list_freq:
            cible_freq = st.selectbox("Cooccurrent (Fréquences)", coocs_list_freq, index=0, key=f"cc_freq_{st.session_state['run_id']}")
            nb_max_freq = st.number_input("Nombre maximum de phrases", min_value=1, max_value=5000, value=200, step=10, key=f"nbmax_freq_{st.session_state['run_id']}")
            lignes_html_freq = []; n_aff = 0
            for sent in sent_spans:
                nlist = iter_tokens_normalises(sent, stopset_cc, pivot_cc, excl_num, excl_1)
                sset = set(nlist)
                if pivot_cc in sset and cible_freq in sset:
                    lignes_html_freq.append(phrase_surface_html(sent, pivot_cc, cible_freq, stopset_cc, excl_num, excl_1))
                    n_aff += 1
                    if n_aff >= int(nb_max_freq):
                        break
            if not lignes_html_freq:
                st.info("Aucune phrase trouvée pour ce cooccurrent.")
            else:
                st.markdown("\n".join(lignes_html_freq), unsafe_allow_html=True)
                doc_html = document_html_kwic(
                    f"Concordancier — Fréquences — pivot = {pivot_cc}, cooccurrent = {cible_freq}",
                    lignes_html_freq
                )
                st.download_button(
                    label="Télécharger le concordancier (Fréquences, HTML)",
                    data=doc_html.encode("utf-8"),
                    file_name=f"concordancier_frequences_{pivot_cc}_{cible_freq}.html",
                    mime="text/html",
                    key=f"dl_kwic_freq_{st.session_state['run_id']}"
                )

        # =====================================
        # 2) LOG-LIKELIHOOD (tableau, nuage, graphe, concordancier)
        # =====================================
        st.markdown("# 2 — Scores log-likelihood")
        st.caption("Score calculé sur les mêmes fenêtres que les fréquences.")
        st.markdown(
            "Le log-likelihood est une mesure statistique qui sert à tester l’indépendance entre deux mots.\n\n"
            "L’idée est de distinguer deux situations :\n"
            "- les cooccurrences qui apparaissent **par simple hasard**, parce que les mots sont fréquents dans le corpus ;\n"
            "- celles qui apparaissent **beaucoup plus souvent que prévu**, et qui révèlent donc une association significative.\n\n"
            "Le log-likelihood est donc une mesure qui permet de faire le tri : "
            "il indique si une cooccurrence est juste due à la fréquence des mots, ou si elle est **anormalement fréquente** et donc révélatrice d’un lien fort.\n\n"
            "En pratique, plus le score est élevé, plus l’association entre les deux mots est intéressante à interpréter."
        )
        df_ll = st.session_state["df_ll"]
        st.dataframe(df_ll, use_container_width=True)
        st.download_button(
            label="Télécharger le CSV (Log-likelihood)",
            data=generer_csv(df_ll).getvalue(),
            file_name="cooccurrences_loglike.csv",
            mime="text/csv",
            key=f"dl_csv_ll_{st.session_state['run_id']}"
        )

        st.markdown("### Nuage de mots — pondéré par le score log-likelihood")
        top_n_ll = st.number_input("Top N (log-likelihood)", min_value=1, max_value=500, value=10, step=1, key=f"top_wc_ll_{st.session_state['run_id']}")
        df_top_ll = df_ll[df_ll["loglike"] > 0].sort_values(["loglike", "fenetres_ensemble"], ascending=[False, False]).head(int(top_n_ll))
        if df_top_ll.empty:
            st.info("Aucun score log-likelihood strictement positif à afficher pour le nuage de mots.")
        else:
            generer_wordcloud(dict(zip(df_top_ll["cooccurrent"], df_top_ll["loglike"])), f"Top {int(top_n_ll)} cooccurrences (log-likelihood)")

        st.markdown("### Graphe interactif — cooccurrences (log-likelihood)")
        n_edges_ll = st.number_input("Nombre d’arêtes (Top-N) – log-likelihood", min_value=1, max_value=200, value=30, step=1, key=f"nedges_ll_{st.session_state['run_id']}")
        poids_ll = dict(zip(df_ll["cooccurrent"], df_ll["loglike"]))
        html_ll = pyvis_reseau_html(pivot_cc, poids_ll, "Réseau — Log-likelihood (G²)", top_n=int(n_edges_ll), syntaxique=False, mode_label="ll", edge_label_size=9)
        st_html(html_ll, height=620, scrolling=True)

        st.markdown("### Concordancier — à partir du log-likelihood")
        coocs_list_ll = list(df_ll["cooccurrent"])
        if coocs_list_ll:
            cible_ll = st.selectbox("Cooccurrent (Log-likelihood)", coocs_list_ll, index=0, key=f"cc_ll_{st.session_state['run_id']}")
            nb_max_ll = st.number_input("Nombre maximum de phrases", min_value=1, max_value=5000, value=200, step=10, key=f"nbmax_ll_{st.session_state['run_id']}")
            lignes_html_ll = []; n_aff = 0
            for sent in sent_spans:
                nlist = iter_tokens_normalises(sent, stopset_cc, pivot_cc, excl_num, excl_1)
                sset = set(nlist)
                if pivot_cc in sset and cible_ll in sset:
                    lignes_html_ll.append(phrase_surface_html(sent, pivot_cc, cible_ll, stopset_cc, excl_num, excl_1))
                    n_aff += 1
                    if n_aff >= int(nb_max_ll):
                        break
            if not lignes_html_ll:
                st.info("Aucune phrase trouvée pour ce cooccurrent (log-likelihood).")
            else:
                st.markdown("\n".join(lignes_html_ll), unsafe_allow_html=True)
                doc_html = document_html_kwic(
                    f"Concordancier — Log-likelihood — pivot = {pivot_cc}, cooccurrent = {cible_ll}",
                    lignes_html_ll
                )
                st.download_button(
                    label="Télécharger le concordancier (Log-likelihood, HTML)",
                    data=doc_html.encode("utf-8"),
                    file_name=f"concordancier_loglike_{pivot_cc}_{cible_ll}.html",
                    mime="text/html",
                    key=f"dl_kwic_ll_{st.session_state['run_id']}"
                )

        # =====================================
        # 3) ANALYSE LEXICO-SYNTAXIQUE (tableau, nuage, graphe, concordancier aligné)
        # =====================================
        st.markdown("# 3 — Cooccurrences lexico-syntaxiques")
        st.caption("Une ligne par cooccurrent et par relation spaCy (nsubj, obj, amod, obl, …).")
        df_syn = st.session_state["df_syn"]
        st.dataframe(df_syn, use_container_width=True)
        st.download_button(
            label="Télécharger le CSV (Syntaxiques)",
            data=generer_csv(df_syn).getvalue(),
            file_name="cooccurrences_syntaxiques.csv",
            mime="text/csv",
            key=f"dl_csv_syn_{st.session_state['run_id']}"
        )

        st.markdown("### Nuage de mots — pondéré par la fréquence (syntaxique)")
        top_n_syn = st.number_input("Top N (syntaxiques)", min_value=1, max_value=500, value=10, step=1, key=f"top_wc_syn_{st.session_state['run_id']}")
        agg_syn = df_syn.groupby("cooccurrent", as_index=False)["frequence"].sum().sort_values("frequence", ascending=False).head(int(top_n_syn))
        generer_wordcloud(dict(zip(agg_syn["cooccurrent"], agg_syn["frequence"])), f"Top {int(top_n_syn)} cooccurrences syntaxiques")

        st.markdown("### Graphe interactif — cooccurrences syntaxiques (label = relation spaCy)")
        n_edges_syn = st.number_input("Nombre d’arêtes (Top-N) – syntaxiques", min_value=1, max_value=200, value=30, step=1, key=f"nedges_syn_{st.session_state['run_id']}")
        poids_syn = dict(((row["cooccurrent"], row["relation_spacy"]), row["frequence"]) for _, row in df_syn.iterrows())
        html_syn = pyvis_reseau_html(pivot_cc, poids_syn, "Réseau — Syntaxique", top_n=int(n_edges_syn), syntaxique=True, edge_label_size=11)
        st_html(html_syn, height=620, scrolling=True)

        st.markdown("### Concordancier Lexico-syntaxique — affichage aligné (relations du pivot)")
        st.caption("Affiche, sous chaque mot relié DIRECTEMENT au pivot, l’étiquette de dépendance spaCy (nsubj, obj, amod, …). Les autres mots restent sans étiquette.")
        if not df_syn.empty:
            cible_syn2 = st.selectbox("Cooccurrent (Syntaxiques, aligné)", sorted(df_syn["cooccurrent"].unique()), index=0, key=f"cc_syn_al_{st.session_state['run_id']}")
            afficher_stop = st.checkbox("Afficher les stopwords dans la grille", value=True, key=f"show_stop_syn_{st.session_state['run_id']}")
            nb_max_syn2 = st.number_input("Nombre maximum de phrases (aligné)", min_value=1, max_value=2000, value=200, step=10, key=f"nbmax_syn_al_{st.session_state['run_id']}")

            lignes_html_syn_al = []; n_aff2 = 0
            for sent, pairs in zip(sent_spans, st.session_state["index_phrase_pairs"]):
                if any((w == cible_syn2) for (w, rel) in pairs):
                    lignes_html_syn_al.append(html_relations_pivot_aligne(sent, pivot=pivot_cc, cible=cible_syn2, afficher_stop=afficher_stop))
                    n_aff2 += 1
                    if n_aff2 >= int(nb_max_syn2):
                        break
            if not lignes_html_syn_al:
                st.info("Aucune phrase trouvée pour ce cooccurrent syntaxique.")
            else:
                st.markdown("\n".join(lignes_html_syn_al), unsafe_allow_html=True)
                doc_html = document_html_kwic(
                    f"Concordancier — Syntaxique (aligné relations pivot) — pivot = {pivot_cc}, cooccurrent = {cible_syn2}",
                    lignes_html_syn_al
                )
                st.download_button(
                    label="Télécharger le concordancier (Syntaxique aligné, HTML)",
                    data=doc_html.encode("utf-8"),
                    file_name=f"concordancier_Lexico_syntaxique_{pivot_cc}_{cible_syn2}.html",
                    mime="text/html",
                    key=f"dl_kwic_syn_al_{st.session_state['run_id']}"
                )

    with ong_lex:
        st.markdown("## Lexique des formes et des relations spaCy observées")
        st.caption("Formes normalisées avec POS ; relations de dépendance rencontrées, avec comptages et exemples issus de votre corpus.")
        st.markdown("### Formes (normalisées) et POS")
        st.dataframe(st.session_state["df_lex_formes"], use_container_width=True)
        st.download_button(
            label="Télécharger le CSV (Formes & POS)",
            data=generer_csv(st.session_state["df_lex_formes"]).getvalue(),
            file_name="lexique_formes_pos.csv",
            mime="text/csv",
            key=f"dl_lex_formes_{st.session_state['run_id']}"
        )

        st.markdown("### Relations de dépendance spaCy observées")
        st.dataframe(st.session_state["df_lex_rel"], use_container_width=True)
        st.download_button(
            label="Télécharger le CSV (Relations spaCy)",
            data=generer_csv(st.session_state["df_lex_rel"]).getvalue(),
            file_name="lexique_relations_spacy.csv",
            mime="text/csv",
            key=f"dl_lex_rel_{st.session_state['run_id']}"
        )

    with ong_exp:
        st.markdown("## Explications spaCy — POS et relations de dépendance")
        st.markdown(
            "Les tableaux suivants récapitulent les **étiquettes POS** et les **relations de dépendance** utilisées par spaCy/UD, avec une définition française fidèle et un exemple minimal.")
        st.markdown("### POS (catégories morpho-syntaxiques)")
        st.dataframe(st.session_state["df_pos_exp"], use_container_width=True)
        st.markdown("### Relations de dépendance (UD/spaCy)")
        st.dataframe(st.session_state["df_dep_exp"], use_container_width=True)

else:
    st.info("Lancez l’analyse pour afficher les tableaux, les nuages de mots, les graphes, les concordanciers, le lexique et les explications.")
