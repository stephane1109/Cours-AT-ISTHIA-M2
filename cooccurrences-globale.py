# python -m streamlit run main.py

# ########## Dépendances
# pip install streamlit pandas spacy matplotlib pyvis scipy wordcloud
# python -m spacy download fr_core_news_md
############


# ================================
# IMPORTS
# ================================
import io
import re
import itertools
import html  # pour html.escape
import math
import numpy as np
import pandas as pd
import streamlit as st
from collections import Counter
from wordcloud import WordCloud
import spacy
import networkx as nx
from pyvis.network import Network
from streamlit.components.v1 import html as st_html
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# ================================
# CHARGEMENT SPACY (FR)
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
    """renvoie l’ensemble des stopwords spaCy si coché, sinon ensemble vide"""
    return set(nlp.Defaults.stop_words) if appliquer_stop else set()

# ================================
# SEGMENTATION PARAGRAPHES
# ================================
def segmenter_paragraphes(texte: str):
    """paragraphe = bloc séparé par au moins une ligne vide"""
    norm = texte.replace("\r\n", "\n").replace("\r", "\n")
    blocs = re.split(r"\n\s*\n+", norm)
    return [b.strip() for b in blocs if b.strip()]

# ================================
# NORMALISATION (APOSTROPHES & FORMES)
# ================================
APOS = {"'", "’"}

def normaliser_apostrophe_joint(token_text: str) -> str:
    """si le token contient une apostrophe, garder la partie à droite (ex. « l'homme » -> « homme »)"""
    if "'" in token_text or "’" in token_text:
        parts = re.split(r"[’']", token_text, maxsplit=1)
        if len(parts) == 2:
            return parts[1]
    return token_text

def iter_tokens_norm_et_carte_global(doc, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """
    renvoie deux listes alignées :
      - norm_list : formes normalisées pour l’analyse globale,
      - spans_list : paires (start_idx, end_excl) en indices de tokens spaCy.
    règles :
      - "c ' est" -> garder "est",
      - "l'homme" -> "homme",
      - minuscule + alphanumérique,
      - filtrage stopwords / nombres / 1_lettre.
    """
    norm_list, spans_list = [], []
    toks = list(doc)
    i, n = 0, len(toks)
    while i < n:
        tok = toks[i]
        raw = tok.text
        low = raw.lower()

        # cas "c ' est" -> garder la partie de droite
        if low.isalpha() and i + 2 < n and toks[i+1].text in APOS:
            droite = toks[i+2].text.lower()
            mot = droite
            if mot.isalnum() and not (exclure_nombres and mot.isdigit()) and not (exclure_monolettre and len(mot) == 1):
                if mot not in stopset:
                    norm_list.append(mot); spans_list.append((i, i+3))
            i += 3
            continue

        # cas "l'homme" -> "homme"
        if ("'" in raw) or ("’" in raw):
            mot = normaliser_apostrophe_joint(raw).lower()
        else:
            mot = low

        if mot.isalnum() and not (exclure_nombres and mot.isdigit()) and not (exclure_monolettre and len(mot) == 1):
            if mot not in stopset:
                norm_list.append(mot); spans_list.append((i, i+1))
        i += 1

    return norm_list, spans_list

def iter_tokens_normalises_global(doc, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """renvoie seulement la séquence normalisée globale"""
    norm, _ = iter_tokens_norm_et_carte_global(doc, stopset, exclure_nombres, exclure_monolettre)
    return norm

# ================================
# FENÊTRES GLOBALES (SANS PIVOT)
# ================================
def fenetres_globales_mots(doc, k: int, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """fenêtres glissantes ±k sur la séquence normalisée (sans pivot)"""
    seq = iter_tokens_normalises_global(doc, stopset, exclure_nombres, exclure_monolettre)
    fenetres = []
    for i in range(len(seq)):
        d = max(0, i - k); f = min(len(seq), i + k + 1)
        S = set(seq[d:f])
        if S:
            fenetres.append(S)
    return fenetres

def fenetres_globales_phrases(doc, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """fenêtres = phrases spaCy (global, sans pivot)"""
    fenetres = []
    for sent in doc.sents:
        seq = iter_tokens_normalises_global(sent, stopset, exclure_nombres, exclure_monolettre)
        if seq:
            fenetres.append(set(seq))
    return fenetres

def fenetres_globales_paragraphes(texte: str, stopset, exclure_nombres: bool, exclure_monolettre: bool):
    """fenêtres = paragraphes (global, sans pivot)"""
    fenetres = []
    for pa in segmenter_paragraphes(texte):
        d = nlp(pa)
        seq = iter_tokens_normalises_global(d, stopset, exclure_nombres, exclure_monolettre)
        if seq:
            fenetres.append(set(seq))
    return fenetres

# ================================
# COOCURRENCES (FRÉQUENCE)
# ================================
def compter_cooc_globales(fenetres):
    """pour chaque fenêtre, incrémente toutes les paires non ordonnées (a,b) avec a<b"""
    compteur = Counter()
    for S in fenetres:
        if len(S) < 2:
            continue
        for a, b in itertools.combinations(sorted(S), 2):
            compteur[(a, b)] += 1
    return compteur

def poids_noeuds_depuis_aretes(paires_freq: dict):
    """somme des poids des arêtes incidentes pour dimensionner/pondérer les nœuds"""
    deg = Counter()
    for (a, b), w in paires_freq.items():
        deg[a] += w
        deg[b] += w
    return deg

def filtrer_par_frequence_au_moins(paires_freq: dict, n: int):
    """garde les arêtes de poids >= n"""
    return {e: w for e, w in paires_freq.items() if w >= n}

# ================================
# PYVIS — GRAPHE GLOBAL (FRÉQUENCE)
# ================================
def _scale_linear(v, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2.0
    return out_min + (out_max - out_min) * ((v - vmin) / (vmax - vmin))

def pyvis_reseau_global_html_couleur(paires_freq: dict, edge_label_size: int = 10):
    """
    réseau global en couleur :
    - couleur des nœuds ~ degré pondéré (somme des fréquences)
    - couleur/largeur des arêtes ~ fréquence de la paire
    - physics activées (ForceAtlas2)
    """
    if not paires_freq:
        return "<div>Aucune cooccurrence globale à afficher (filtre trop strict ?).</div>"

    # degrés pondérés
    deg = Counter()
    for (a, b), w in paires_freq.items():
        deg[a] += float(w); deg[b] += float(w)
    deg_vals = list(deg.values()) or [1.0]
    dmin, dmax = min(deg_vals), max(deg_vals)

    wvals = [float(w) for w in paires_freq.values()] or [1.0]
    wmin, wmax = min(wvals), max(wvals)

    net = Network(height="900px", width="100%", directed=False, notebook=False, cdn_resources="in_line")
    net.set_options("""{
      "interaction": {"dragNodes": true, "dragView": true, "zoomView": true},
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.005,
          "springLength": 150,
          "springConstant": 0.08,
          "avoidOverlap": 1.0
        },
        "stabilization": {"enabled": true, "iterations": 300, "fit": true, "updateInterval": 25}
      },
      "edges": {"smooth": false},
      "nodes": {"shape": "dot", "scaling": {"min": 8, "max": 60}}
    }""")

    # nœuds
    vus = set()
    for (a, b), _w in paires_freq.items():
        for n_ in (a, b):
            if n_ in vus:
                continue
            vus.add(n_)
            taille = _scale_linear(deg[n_], dmin, dmax, 10, 58)
            t = 0.0 if dmax == dmin else (deg[n_] - dmin) / (dmax - dmin)
            col = mcolors.to_hex(cm.viridis(t))
            net.add_node(n_, label=n_, title=f"{n_} (degré pondéré={deg[n_]:.0f})", size=taille, color=col)

    # arêtes
    for (a, b), w in paires_freq.items():
        w = float(w)
        width = _scale_linear(w, wmin, wmax, 1.2, 6.5)
        t = 0.0 if wmax == wmin else (w - wmin) / (wmax - wmin)
        ecol = mcolors.to_hex(cm.plasma(t))
        net.add_edge(a, b, value=w, width=width, color=ecol, label=str(int(w)),
                     title=f"fréquence={int(w)}", font={"size": edge_label_size})

    return net.generate_html()

# ================================
# CONCORDANCIERS (FONCTIONS)
# ================================
def surligner_phrase_paire(sent, w1: str, w2: str):
    """surligne w1 et w2 dans la phrase surface"""
    toks = list(sent)
    out = []
    css = (
        "<style>"
        ".kwic-sent{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Noto Sans','Helvetica Neue',Arial;"
        "line-height:1.6;margin:6px 0}"
        ".w1{background:#e63946;color:#fff;border-radius:3px;padding:0 3px}"
        ".w2{background:#1d3557;color:#fff;border-radius:3px;padding:0 3px}"
        "</style>"
    )
    for tok in toks:
        surf = html.escape(tok.text_with_ws)
        norm = normaliser_apostrophe_joint(tok.text).lower()
        if norm == w1:
            out.append(f"<span class='w1'>{surf}</span>")
        elif norm == w2:
            out.append(f"<span class='w2'>{surf}</span>")
        else:
            out.append(surf)
    return css + f"<div class='kwic-sent'>{''.join(out).strip()}</div>"

def kwic_mots_pm_k(doc, stopset, exclure_nombres, exclure_monolettre, w1: str, w2: str, k: int, marge: int = 5):
    """concordancier « mots (±k) » : extrait un segment surface lorsque w1 et w2 sont à ≤k positions"""
    norm, spans = iter_tokens_norm_et_carte_global(doc, stopset, exclure_nombres, exclure_monolettre)
    toks = list(doc)
    pos_w1 = [i for i, w in enumerate(norm) if w == w1]
    pos_w2 = [i for i, w in enumerate(norm) if w == w2]
    if not pos_w1 or not pos_w2:
        return []

    deja = set()
    lignes = []

    for i in pos_w1:
        for j in pos_w2:
            if abs(i - j) <= k:
                t_start = min(spans[i][0], spans[j][0])
                t_end   = max(spans[i][1], spans[j][1])
                s = max(0, t_start - marge)
                e = min(len(toks), t_end + marge)
                key = (s, e)
                if key in deja:
                    continue
                deja.add(key)

                head_i = max(spans[i][1]-1, spans[i][0])
                head_j = max(spans[j][1]-1, spans[j][0])

                css = (
                    "<style>"
                    ".kwic-sent{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Noto Sans','Helvetica Neue',Arial;"
                    "line-height:1.6;margin:6px 0}"
                    ".w1{background:#e63946;color:#fff;border-radius:3px;padding:0 3px}"
                    ".w2{background:#1d3557;color:#fff;border-radius:3px;padding:0 3px}"
                    "</style>"
                )
                out = []
                for idx in range(s, e):
                    surf = html.escape(toks[idx].text_with_ws)
                    if idx == head_i:
                        out.append(f"<span class='w1'>{surf}</span>")
                    elif idx == head_j:
                        out.append(f"<span class='w2'>{surf}</span>")
                    else:
                        out.append(surf)
                lignes.append(css + f"<div class='kwic-sent'>{''.join(out).strip()}</div>")
    return lignes

def surligner_paragraphe_paire(pa_texte: str, w1: str, w2: str):
    """surligne w1 et w2 dans un paragraphe entier (surface)"""
    d = nlp(pa_texte)
    toks = list(d)
    out = []
    css = (
        "<style>"
        ".kwic-sent{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Noto Sans','Helvetica Neue',Arial;"
        "line-height:1.6;margin:6px 0}"
        ".w1{background:#e63946;color:#fff;border-radius:3px;padding:0 3px}"
        ".w2{background:#1d3557;color:#fff;border-radius:3px;padding:0 3px}"
        "</style>"
    )
    for tok in toks:
        surf = html.escape(tok.text_with_ws)
        norm = normaliser_apostrophe_joint(tok.text).lower()
        if norm == w1:
            out.append(f"<span class='w1'>{surf}</span>")
        elif norm == w2:
            out.append(f"<span class='w2'>{surf}</span>")
        else:
            out.append(surf)
    return css + f"<div class='kwic-sent'>{''.join(out).strip()}</div>"

def document_html_kwic(titre: str, sections_html):
    """document html autonome pour télécharger un concordancier global"""
    style = (
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,'Noto Sans','Helvetica Neue',Arial;line-height:1.6;padding:12px}"
        "h2{margin:18px 0 8px 0}"
        ".kwic-sent{margin:6px 0}"
        "</style>"
    )
    corps = "\n".join(sections_html)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(titre)}</title>{style}</head><body>"
        f"<h1>{html.escape(titre)}</h1>"
        f"{corps}</body></html>"
    )

# ================================
# OUTILS
# ================================
def generer_csv(df: pd.DataFrame):
    """retourne un tampon csv téléchargeable"""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

def generer_nuage_mots(degre_pondere: dict, titre: str):
    """nuage de mots : taille = degré pondéré (somme des fréquences de cooccurrence)"""
    freq_pos = {w: float(v) for w, v in degre_pondere.items() if v and v > 0}
    if not freq_pos:
        st.info("Nuage non généré : aucune valeur strictement positive.")
        return
    wc = WordCloud(width=900, height=450, background_color="white").generate_from_frequencies(freq_pos)
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear"); plt.axis("off"); plt.title(titre)
    st.pyplot(fig)
    # téléchargement PNG
    buf_png = io.BytesIO()
    wc.to_image().save(buf_png, format="PNG")
    buf_png.seek(0)
    st.session_state["nuage_png"] = buf_png.getvalue()

def generer_png_graphe_stat(paires_freq: dict, largeur_px: int = 1600, hauteur_px: int = 1000, dpi: int = 100) -> bytes:
    """PNG statique du graphe pondéré (fréquence)"""
    if not paires_freq:
        return b""
    G = nx.Graph()
    for (a, b), w in paires_freq.items():
        G.add_edge(a, b, weight=float(w))

    # degré pondéré (somme des poids)
    deg_p = {n: float(sum(d["weight"] for _u, _v, d in G.edges(n, data=True))) for n in G.nodes()}
    if not deg_p:
        return b""

    # positions (spring) pondérées
    n = max(G.number_of_nodes(), 1)
    k_dist = 1.2 / (n ** 0.5)
    pos = nx.spring_layout(G, weight="weight", seed=42, k=k_dist, iterations=200)

    # échelles
    dvals = list(deg_p.values())
    dmin, dmax = min(dvals), max(dvals)
    def scale(val, vmin, vmax, out_min, out_max):
        if vmax == vmin:
            return (out_min + out_max) / 2.0
        return out_min + (out_max - out_min) * ((val - vmin) / (vmax - vmin))

    # figure 1600×1000
    fig_w = largeur_px / dpi
    fig_h = hauteur_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")

    # couleurs/tailles des nœuds
    ncolors = [cm.viridis(scale(deg_p[n], dmin, dmax, 0.0, 1.0)) for n in G.nodes()]
    nsizes = [scale(deg_p[n], dmin, dmax, 300, 3000) for n in G.nodes()]

    # arêtes
    wvals = [d["weight"] for _u, _v, d in G.edges(data=True)]
    wmin, wmax = (min(wvals), max(wvals)) if wvals else (1.0, 1.0)
    ewidths = [scale(w, wmin, wmax, 0.8, 5.0) for w in wvals]
    ecolors = [cm.plasma(scale(w, wmin, wmax, 0.0, 1.0)) for w in wvals]

    nx.draw_networkx_edges(G, pos, ax=ax, width=ewidths, edge_color=ecolors, alpha=0.9)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=nsizes, node_color=ncolors)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=14)

    buf = io.BytesIO()
    fig.tight_layout(pad=0)
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ================================
# NOUVEAU — LOG-LIKELIHOOD (DUNNING 1993)
# ================================
def compter_presence_par_fenetre(fenetres):
    """compte, pour chaque mot, dans combien de fenêtres il apparaît au moins une fois"""
    c = Counter()
    for S in fenetres:
        for w in S:
            c[w] += 1
    return c

def tableau_contingence(a: str, b: str, pres_counts: dict, k11: int, N: int):
    """
    Table de contingence sur les fenêtres :
        k11 = fenêtres contenant a et b
        k12 = fenêtres contenant a sans b
        k21 = fenêtres contenant b sans a
        k22 = fenêtres contenant ni a ni b
    """
    k12 = pres_counts.get(a, 0) - k11
    k21 = pres_counts.get(b, 0) - k11
    k22 = N - k11 - k12 - k21
    k12 = max(0, k12); k21 = max(0, k21); k22 = max(0, k22)
    return k11, k12, k21, k22

def _xlogx(x: float) -> float:
    """x*log(x) (défini à 0 pour x=0)"""
    return x * math.log(x) if x > 0 else 0.0

def log_likelihood(k11, k12, k21, k22) -> float:
    """Score G² de Dunning : plus il est grand, plus l'association est forte/surprenante."""
    row1 = k11 + k12
    row2 = k21 + k22
    col1 = k11 + k21
    col2 = k12 + k22
    N = row1 + row2
    return 2.0 * (
        _xlogx(k11) + _xlogx(k12) + _xlogx(k21) + _xlogx(k22)
        - _xlogx(row1) - _xlogx(row2) - _xlogx(col1) - _xlogx(col2)
        + _xlogx(N)
    )

def calculer_llr_paires(paires_freq: dict, fenetres) -> dict:
    """calcule le log-likelihood pour toutes les paires observées au moins une fois"""
    if not paires_freq:
        return {}
    pres = compter_presence_par_fenetre(fenetres)
    N = len(fenetres)
    llr = {}
    for (a, b), k11 in paires_freq.items():
        k11 = int(k11)
        k11, k12, k21, k22 = tableau_contingence(a, b, pres, k11, N)
        llr[(a, b)] = log_likelihood(k11, k12, k21, k22)
    return llr

def pyvis_graphe_likelihood_html(llr_scores: dict, seuil_llr: float = 0.0):
    """graphe PyVis où les arêtes sont pondérées par le log-likelihood"""
    data = {e: s for e, s in llr_scores.items() if s >= seuil_llr}
    if not data:
        return "<div>Aucune arête après application du seuil de likelihood.</div>"

    # degré « pondéré LLR » pour les nœuds
    deg = Counter()
    for (a, b), s in data.items():
        deg[a] += s; deg[b] += s
    dmin, dmax = (min(deg.values()), max(deg.values()))

    smin, smax = (min(data.values()), max(data.values()))
    net = Network(height="900px", width="100%", directed=False, notebook=False, cdn_resources="in_line")
    net.set_options("""{
      "interaction": {"dragNodes": true, "dragView": true, "zoomView": true},
      "physics": {
        "enabled": true,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.005, "springLength": 150, "springConstant": 0.08, "avoidOverlap": 1.0},
        "stabilization": {"enabled": true, "iterations": 300, "fit": true, "updateInterval": 25}
      },
      "edges": {"smooth": false},
      "nodes": {"shape": "dot", "scaling": {"min": 8, "max": 60}}
    }""")

    vus = set()
    for (a, b), _ in data.items():
        for n_ in (a, b):
            if n_ in vus:
                continue
            vus.add(n_)
            taille = _scale_linear(deg[n_], dmin, dmax, 10, 58)
            t = 0.0 if dmax == dmin else (deg[n_] - dmin) / (dmax - dmin)
            col = mcolors.to_hex(cm.viridis(t))
            net.add_node(n_, label=n_, title=f"{n_} (somme LLR={deg[n_]:.2f})", size=taille, color=col)

    for (a, b), s in data.items():
        width = _scale_linear(s, smin, smax, 1.2, 6.5)
        ecol = mcolors.to_hex(cm.inferno(_scale_linear(s, smin, smax, 0.0, 1.0)))
        net.add_edge(a, b, value=s, width=width, color=ecol, label=f"{s:.2f}", title=f"LLR={s:.2f}", font={"size": 10})

    return net.generate_html()

# ================================
# INTERFACE — PARAMÈTRES ET EXPLICATIONS
# ================================
st.set_page_config(page_title="Cooccurrences — Graphe, Likelihood, Nuage, Concordancier")
st.title("Cooccurrences : graphe, likelihood, nuage pondéré et concordancier")

st.markdown(
    "Cette application calcule des cooccurrences **sans mot pivot**. "
    "Une cooccurrence est comptée quand deux mots apparaissent ensemble dans la **même fenêtre**. "
    "Fenêtres possibles : « Mots (±k) » (glissante), « Phrase » (spaCy) et « Paragraphe » (séparé par une ligne vide)."
)
st.markdown(
    "Le **filtrage par fréquence N** conserve les paires vues **au moins N fois**. "
    "Le graphe affiche toutes les arêtes retenues. Le **concordancier** liste les occurrences ; "
    "l’aperçu est **limité aux 10 premières phrases** et le **téléchargement** fournit l’intégralité."
)

uploaded = st.file_uploader("Fichier texte (.txt)", type=["txt"])
texte_libre = st.text_area("Ou collez votre texte ici", height=220, placeholder="Collez votre corpus ici…")

st.subheader("Fenêtre de contexte")
fenetre = st.selectbox("Type de fenêtre", ["Mots (±k)", "Phrase", "Paragraphe"])
k = 5
if fenetre == "Mots (±k)":
    k = st.number_input("Paramètre k (taille de la fenêtre en mots)", min_value=1, max_value=10000, value=5, step=1)

st.subheader("Options de nettoyage")
appliquer_stop = st.checkbox("Appliquer les stopwords (spaCy)", value=True)
exclure_nombres = st.checkbox("Exclure les nombres", value=True)
exclure_monolettre = st.checkbox("Exclure les mots d’une seule lettre", value=True)

st.subheader("Filtrage par fréquence")
n_filtre = st.number_input("Fréquence minimale N (au moins N)", min_value=1, max_value=100000, value=2, step=1)

# Paramètres likelihood
st.subheader("Paramètres likelihood")
max_mots_matrice = st.number_input("Nombre max. de mots dans la matrice LLR", min_value=5, max_value=200, value=50, step=1)
seuil_llr_graphe = st.number_input("Seuil minimal LLR pour le graphe (arêtes ≥ seuil)", min_value=0.0, max_value=1e9, value=5.0, step=1.0)

if st.button("Calculer les cooccurrences"):
    # Corpus
    texte = uploaded.read().decode("utf-8", errors="ignore") if uploaded else texte_libre
    if not texte or not texte.strip():
        st.error("Veuillez fournir un texte.")
        st.stop()

    # Préparation
    stopset = construire_stopwords(appliquer_stop)
    doc = nlp(texte)

    # Fenêtres globales
    if fenetre == "Mots (±k)":
        fenetres = fenetres_globales_mots(doc, int(k), stopset, exclure_nombres, exclure_monolettre)
    elif fenetre == "Phrase":
        fenetres = fenetres_globales_phrases(doc, stopset, exclure_nombres, exclure_monolettre)
    else:
        fenetres = fenetres_globales_paragraphes(texte, stopset, exclure_nombres, exclure_monolettre)

    # Comptage paires (brut) puis filtrage « au moins N »
    paires = compter_cooc_globales(fenetres)
    paires = filtrer_par_frequence_au_moins(paires, int(n_filtre))

    # Table des paires retenues
    df_paires = pd.DataFrame(
        [(a, b, int(w)) for (a, b), w in sorted(paires.items(), key=lambda x: x[1], reverse=True)],
        columns=["mot1", "mot2", "frequence"]
    )
    st.subheader("Table des cooccurrences retenues")
    if df_paires.empty:
        st.info("Aucune paire après filtrage. Assouplissez N ou modifiez les options.")
        st.stop()
    st.dataframe(df_paires.head(3000), use_container_width=True)
    st.download_button(
        label="Télécharger le CSV (paires filtrées)",
        data=generer_csv(df_paires).getvalue(),
        file_name="cooccurrences_filtrees.csv",
        mime="text/csv"
    )

    # Statistiques simples
    noeuds = set()
    for (a, b) in paires.keys():
        noeuds.add(a); noeuds.add(b)
    E = len(paires); V = len(noeuds)
    densite = 0.0 if V < 2 else (2 * E) / (V * (V - 1))
    st.write(f"Nœuds : {V} — Arêtes : {E} — Densité ≈ {densite:.3f}")

    # Graphe (fréquence) — PyVis
    st.subheader("Graphe global (pondéré par la fréquence)")
    st.markdown(
        "Le **degré pondéré** d’un mot est la **somme des fréquences** de ses arêtes. "
        "Il dimensionne les nœuds ci-dessous ; la couleur/largeur des arêtes reflète f(a,b)."
    )
    html_global = pyvis_reseau_global_html_couleur(paires, edge_label_size=10)
    st_html(html_global, height=900, scrolling=True)
    st.download_button(
        label="Télécharger le graphe (HTML)",
        data=html_global.encode("utf-8"),
        file_name="graphe_cooccurrences.html",
        mime="text/html"
    )

    # PNG statique
    png_graphe = generer_png_graphe_stat(paires, largeur_px=1600, hauteur_px=1000, dpi=100)
    if png_graphe:
        st.download_button(
            label="Télécharger le graphe (PNG 1600px)",
            data=png_graphe,
            file_name="graphe_cooccurrences.png",
            mime="image/png"
        )

    # Nuage pondéré
    st.subheader("Nuage pondéré des nœuds")
    st.markdown("La taille de chaque mot est proportionnelle à la somme des fréquences de ses cooccurrences.")
    deg_pond = poids_noeuds_depuis_aretes(paires)
    generer_nuage_mots(deg_pond, "Importance des nœuds (degré pondéré)")
    if "nuage_png" in st.session_state:
        st.download_button(
            label="Télécharger le nuage (PNG)",
            data=st.session_state["nuage_png"],
            file_name="nuage_cooccurrences.png",
            mime="image/png"
        )

    # ============================================================
    # CONCORDANCIER (FRÉQUENCE) — APERÇU 10 PHRASES + TÉLÉCHARGEMENT COMPLET
    # ============================================================
    st.subheader("Concordancier global (fréquence)")
    st.markdown(
        "Aperçu limité aux **10 premières phrases** (toutes paires confondues). "
        "Le bouton ci-dessous permet de **télécharger le concordancier complet**."
    )

    sections_full = []        # pour le document complet
    all_lines_for_preview = []  # (titre_pair, ligne_html) pour limiter à 10

    paires_triees = sorted(paires.items(), key=lambda x: x[1], reverse=True)

    if fenetre == "Phrase":
        sent_infos = [(sent, set(iter_tokens_normalises_global(sent, stopset, exclure_nombres, exclure_monolettre))) for sent in doc.sents]
        for (w1, w2), w in paires_triees:
            titre_pair = f"{html.escape(w1)} — {html.escape(w2)} (f={int(w)})"
            lignes = []
            for sent, sset in sent_infos:
                if w1 in sset and w2 in sset:
                    h = surligner_phrase_paire(sent, w1, w2)
                    lignes.append(h)
                    all_lines_for_preview.append((titre_pair, h))
            if lignes:
                sections_full.append(f"<h2>{titre_pair}</h2>" + "\n".join(lignes))

    elif fenetre == "Mots (±k)":
        for (w1, w2), w in paires_triees:
            titre_pair = f"{html.escape(w1)} — {html.escape(w2)} (f={int(w)})"
            lignes = kwic_mots_pm_k(doc, stopset, exclure_nombres, exclure_monolettre, w1=w1, w2=w2, k=int(k), marge=5)
            for h in lignes:
                all_lines_for_preview.append((titre_pair, h))
            if lignes:
                sections_full.append(f"<h2>{titre_pair}</h2>" + "\n".join(lignes))

    else:  # Paragraphe
        paras = segmenter_paragraphes(texte)
        para_infos = []
        for pa in paras:
            d = nlp(pa)
            sset = set(iter_tokens_normalises_global(d, stopset, exclure_nombres, exclure_monolettre))
            para_infos.append((pa, sset))
        for (w1, w2), w in paires_triees:
            titre_pair = f"{html.escape(w1)} — {html.escape(w2)} (f={int(w)})"
            lignes = []
            for pa, sset in para_infos:
                if w1 in sset and w2 in sset:
                    h = surligner_paragraphe_paire(pa, w1, w2)
                    lignes.append(h)
                    all_lines_for_preview.append((titre_pair, h))
            if lignes:
                sections_full.append(f"<h2>{titre_pair}</h2>" + "\n".join(lignes))

    # aperçu 10 lignes max
    if not all_lines_for_preview:
        st.info("Aucun extrait trouvé pour le concordancier avec les options actuelles.")
    else:
        apercu_parts = []
        for titre_pair, ligne in all_lines_for_preview[:10]:
            apercu_parts.append(f"<h3>{titre_pair}</h3>{ligne}")
        st.markdown("\n".join(apercu_parts), unsafe_allow_html=True)

        # téléchargement du document complet
        doc_html_freq = document_html_kwic("Concordancier global — cooccurrences (fréquence)", sections_full)
        st.download_button(
            label="Télécharger le concordancier complet (HTML)",
            data=doc_html_freq.encode("utf-8"),
            file_name="concordancier_frequence_complet.html",
            mime="text/html"
        )

    # ============================================================
    # SECTION LIKELIHOOD : EXPLICATION + TABLE + MATRICE + GRAPHE + CONCORDANCIER
    # ============================================================
    st.subheader("Approche par log-likelihood (Dunning)")
    st.markdown(
        "Le **log-likelihood** mesure à quel point l’association entre deux mots est **surprenante** au regard d’une "
        "hypothèse d’indépendance. Pour chaque paire (w1, w2), on construit une **table de contingence** sur les fenêtres : "
        "k11 = fenêtres avec w1 et w2 ; k12 = w1 sans w2 ; k21 = w2 sans w1 ; k22 = ni w1 ni w2. "
        "Le score G² est ensuite calculé ; **plus G² est grand, plus l’association est forte**."
    )

    # scores LLR
    llr_scores = calculer_llr_paires(paires, fenetres)
    if not llr_scores:
        st.info("Aucun score de likelihood n’a pu être calculé.")
        st.stop()

    # Tableau trié par LLR décroissant
    df_llr = pd.DataFrame(
        [(a, b, int(df_paires.loc[(df_paires['mot1']==a) & (df_paires['mot2']==b), 'frequence'].values[0]), s)
         for (a, b), s in sorted(llr_scores.items(), key=lambda x: x[1], reverse=True)],
        columns=["mot1", "mot2", "frequence", "log_likelihood"]
    )
    st.markdown("Table des paires triées par **log-likelihood décroissant**.")
    st.dataframe(df_llr.head(3000), use_container_width=True)
    st.download_button(
        label="Télécharger le CSV (paires + LLR)",
        data=generer_csv(df_llr).getvalue(),
        file_name="cooccurrences_likelihood.csv",
        mime="text/csv"
    )

    # Matrice (heatmap) LLR
    st.subheader("Matrice de likelihood (heatmap)")
    st.markdown(
        "Matrice symétrique où la case (a,b) contient le score **G²**. "
        "On limite l’affichage aux *max_mots_matrice* premiers mots présents dans les arêtes filtrées."
    )
    voc = sorted(list(noeuds))[: int(max_mots_matrice)]
    idx = {w: i for i, w in enumerate(voc)}
    M = np.zeros((len(voc), len(voc)), dtype=float)
    for (a, b), s in llr_scores.items():
        if a in idx and b in idx:
            i, j = idx[a], idx[b]
            M[i, j] = s
            M[j, i] = s

    fig, ax = plt.subplots(figsize=(min(14, 0.3*len(voc)+5), min(10, 0.3*len(voc)+3)))
    im = ax.imshow(M, cmap="magma")
    ax.set_xticks(np.arange(len(voc))); ax.set_yticks(np.arange(len(voc)))
    ax.set_xticklabels(voc, rotation=90, fontsize=9)
    ax.set_yticklabels(voc, fontsize=9)
    ax.set_title("Matrice log-likelihood")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    st.pyplot(fig)

    # Graphe basé sur le likelihood
    st.subheader("Graphe basé sur le likelihood")
    st.markdown(
        "Arêtes pondérées par **G²** ; les nœuds sont dimensionnés par la **somme des scores LLR** de leurs arêtes."
    )
    html_llr = pyvis_graphe_likelihood_html(llr_scores, seuil_llr=float(seuil_llr_graphe))
    st_html(html_llr, height=900, scrolling=True)
    st.download_button(
        label="Télécharger le graphe likelihood (HTML)",
        data=html_llr.encode("utf-8"),
        file_name="graphe_likelihood.html",
        mime="text/html"
    )

    # Concordancier (likelihood) — aperçu 10 lignes + téléchargement complet
    st.subheader("Concordancier — meilleures associations (log-likelihood)")
    st.markdown(
        "Aperçu limité aux **10 premières phrases** (toutes paires confondues), triées par **force d’association** (G²). "
        "Le téléchargement fournit le **concordancier complet**."
    )
    sections_llr_full = []
    all_lines_llr_preview = []

    paires_llr_triees = sorted(llr_scores.items(), key=lambda x: x[1], reverse=True)

    if fenetre == "Phrase":
        sent_infos = [(sent, set(iter_tokens_normalises_global(sent, stopset, exclure_nombres, exclure_monolettre))) for sent in doc.sents]
        for (w1, w2), s in paires_llr_triees:
            titre_pair = f"{html.escape(w1)} — {html.escape(w2)} (LLR={s:.2f})"
            lignes = []
            for sent, sset in sent_infos:
                if w1 in sset and w2 in sset:
                    h = surligner_phrase_paire(sent, w1, w2)
                    lignes.append(h)
                    all_lines_llr_preview.append((titre_pair, h))
            if lignes:
                sections_llr_full.append(f"<h2>{titre_pair}</h2>" + "\n".join(lignes))

    elif fenetre == "Mots (±k)":
        for (w1, w2), s in paires_llr_triees:
            titre_pair = f"{html.escape(w1)} — {html.escape(w2)} (LLR={s:.2f})"
            lignes = kwic_mots_pm_k(doc, stopset, exclure_nombres, exclure_monolettre, w1=w1, w2=w2, k=int(k), marge=5)
            for h in lignes:
                all_lines_llr_preview.append((titre_pair, h))
            if lignes:
                sections_llr_full.append(f"<h2>{titre_pair}</h2>" + "\n".join(lignes))

    else:  # Paragraphe
        paras = segmenter_paragraphes(texte)
        para_infos = []
        for pa in paras:
            d = nlp(pa)
            sset = set(iter_tokens_normalises_global(d, stopset, exclure_nombres, exclure_monolettre))
            para_infos.append((pa, sset))
        for (w1, w2), s in paires_llr_triees:
            titre_pair = f"{html.escape(w1)} — {html.escape(w2)} (LLR={s:.2f})"
            lignes = []
            for pa, sset in para_infos:
                if w1 in sset and w2 in sset:
                    h = surligner_paragraphe_paire(pa, w1, w2)
                    lignes.append(h)
                    all_lines_llr_preview.append((titre_pair, h))
            if lignes:
                sections_llr_full.append(f"<h2>{titre_pair}</h2>" + "\n".join(lignes))

    if not all_lines_llr_preview:
        st.info("Aucun extrait trouvé pour le concordancier likelihood avec les options actuelles.")
    else:
        apercu_llr = []
        for titre_pair, ligne in all_lines_llr_preview[:10]:
            apercu_llr.append(f"<h3>{titre_pair}</h3>{ligne}")
        st.markdown("\n".join(apercu_llr), unsafe_allow_html=True)

        doc_html_llr = document_html_kwic("Concordancier — log-likelihood (complet)", sections_llr_full)
        st.download_button(
            label="Télécharger le concordancier likelihood complet (HTML)",
            data=doc_html_llr.encode("utf-8"),
            file_name="concordancier_likelihood_complet.html",
            mime="text/html"
        )
