# pip install pandas streamlit
# python -m streamlit run main.py

"""
Fonctionnalités :
- Prétraitement (minuscules, ponctuation, liste minimale de mots-vides)
- Matrice n°1 : TDM binaire (documents, 0/1)
- Matrice n°2 : TDM "fréquence des occurrences" (termes en lignes, documents en colonnes, comptes bruts)
- Énumération de toutes les bipartitions (Groupe 1 - Groupe 2) avec toutes les matrices regroupées possibles
- Pour chaque bipartition :
    • matrices de présence/absence par groupe (documents en lignes) [préservées pour cohérence pédagogique]
    • matrice REGROUPÉE avec GROUPES EN LIGNES (Classe_1, Classe_2) et TERMES EN COLONNES
    • calcul du χ² global et tableau des contributions par terme
- Export CSV des matrices affichées
"""

import re
import itertools
import pandas as pd
import streamlit as st
from collections import Counter, OrderedDict

# La toute première commande Streamlit
st.set_page_config(page_title="Comprendre la CHD — test du χ²", layout="wide")

# Constantes d'affichage
K_TOP = 10  # nombre de termes affichés dans les tableaux de contributions

# -------------------- Stopwords NLTK (français uniquement) --------------------

@st.cache_resource
def charger_mots_vides_nltk() -> set:
    """
    Charger les mots vides français depuis NLTK.
    En cas d'absence de NLTK, afficher une erreur et arrêter l'application.
    """
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            mots = set(stopwords.words("french"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            mots = set(stopwords.words("french"))
        return {w.lower() for w in mots}
    except ImportError:
        st.error("NLTK n'est pas installé. Veuillez exécuter : pip install nltk")
        st.stop()

MOTS_VIDES = charger_mots_vides_nltk()

# -------------------- Prétraitement et TDM --------------------

def normaliser_apostrophes(texte: str) -> str:
    """Remplacer les apostrophes typographiques et simples par des espaces pour séparer correctement les mots."""
    return re.sub(r"[’']", " ", texte)

def nettoyer_tokeniser(texte: str, supprimer_mots_vides: bool = True) -> list:
    """
    Mettre en minuscules, retirer la ponctuation (en conservant lettres accentuées et tirets), puis tokeniser.
    Si supprimer_mots_vides=True, retirer les mots vides issus de NLTK uniquement.
    """
    t = normaliser_apostrophes(texte.lower())
    t = re.sub(r"[^a-zàâäçéèêëîïôöùûüÿœ\- ]", " ", t)
    tokens = [tok for tok in t.split() if tok]
    if supprimer_mots_vides:
        tokens = [tok for tok in tokens if tok not in MOTS_VIDES]
    return tokens

def construire_tdm(corpus: list, supprimer_mots_vides: bool = True) -> pd.DataFrame:
    """
    Matrice terme-document 'classique' (comptes bruts) pour le calcul :
    lignes = termes, colonnes = D1..Dn. Compte bien les occurrences répétées d'un terme par document.
    """
    tokens_par_doc = [nettoyer_tokeniser(doc, supprimer_mots_vides=supprimer_mots_vides) for doc in corpus]
    vocab = sorted({tok for doc in tokens_par_doc for tok in doc})
    compteurs = [Counter(doc) for doc in tokens_par_doc]
    donnees = OrderedDict()
    for terme in vocab:
        donnees[terme] = [compteurs[i].get(terme, 0) for i in range(len(corpus))]
    df = pd.DataFrame(donnees).T
    df.columns = [f"D{i+1}" for i in range(len(corpus))]
    return df

def construire_tdm_binaire_docs_en_lignes(corpus: list, supprimer_mots_vides: bool = True) -> pd.DataFrame:
    """Matrice binaire : colonnes = termes, lignes = D1..Dn, valeurs 0/1."""
    tokens_par_doc = [nettoyer_tokeniser(doc, supprimer_mots_vides=supprimer_mots_vides) for doc in corpus]
    vocab = sorted({tok for doc in tokens_par_doc for tok in doc})
    lignes = []
    for doc_tokens in tokens_par_doc:
        ens = set(doc_tokens)
        lignes.append([1 if terme in ens else 0 for terme in vocab])
    df = pd.DataFrame(lignes, columns=vocab)
    df.index = [f"D{i+1}" for i in range(len(corpus))]
    return df

# -------------------- Regroupement et χ² --------------------

def regrouper_par_classe(tdm_termes_lignes: pd.DataFrame, etiquettes: list,
                         noms_classes=("Classe_1","Classe_2")) -> pd.DataFrame:
    """
    Regrouper les colonnes documents d'une TDM 'termes en lignes' selon un vecteur d'étiquettes 0/1.
    Retour : V × 2 (lignes = termes, colonnes = classes) avec comptes bruts par classe.
    """
    if len(etiquettes) != tdm_termes_lignes.shape[1]:
        raise ValueError("Le nombre d'étiquettes doit correspondre au nombre de documents.")
    cols_1 = [i for i, e in enumerate(etiquettes) if e == 0]
    cols_2 = [i for i, e in enumerate(etiquettes) if e == 1]
    if len(cols_1) == 0 or len(cols_2) == 0:
        raise ValueError("Chaque classe doit contenir au moins un document.")
    somme_1 = tdm_termes_lignes.iloc[:, cols_1].sum(axis=1)
    somme_2 = tdm_termes_lignes.iloc[:, cols_2].sum(axis=1)
    regroupée_Vx2 = pd.DataFrame({noms_classes[0]: somme_1, noms_classes[1]: somme_2})
    return regroupée_Vx2

def fabriquer_groupes_en_lignes(table_Vx2: pd.DataFrame) -> pd.DataFrame:
    """Produire la matrice REGROUPÉE"""
    return table_Vx2.T

def calculer_chi2_global(table_Vx2: pd.DataFrame) -> float:
    """Calculer le χ² global d'indépendance pour une table V × 2 (termes × classes)."""
    O = table_Vx2.values
    total = O.sum()
    if total == 0:
        return 0.0
    somme_lignes = O.sum(axis=1, keepdims=True)
    somme_colonnes = O.sum(axis=0, keepdims=True)
    E = (somme_lignes @ somme_colonnes) / total
    masque = E > 0
    chi2 = ((O - E)**2 / E)[masque].sum()
    return float(chi2)

def contributions_par_terme(table_Vx2: pd.DataFrame) -> pd.DataFrame:
    """Contributions au χ² par terme. Colonnes : Classe_1, Classe_2, contrib_totale (pas d'écarts ni de côté)."""
    O = table_Vx2.values
    termes = table_Vx2.index
    classes = list(table_Vx2.columns)
    total = O.sum()
    if total == 0:
        return pd.DataFrame(index=termes)
    somme_lignes = O.sum(axis=1, keepdims=True)
    somme_colonnes = O.sum(axis=0, keepdims=True)
    E = (somme_lignes @ somme_colonnes) / total
    contrib_cellules = pd.DataFrame(((O - E)**2) / E, index=termes, columns=classes).fillna(0.0)
    contrib_totale = contrib_cellules.sum(axis=1).rename("contrib_totale")
    res = pd.concat([contrib_cellules, contrib_totale], axis=1).sort_values("contrib_totale", ascending=False)
    return res

def tableau_top_contribs_sans_ecart(contribs: pd.DataFrame, k: int = K_TOP) -> pd.DataFrame:
    """Préparer le tableau des contributions en ne gardant que 'contrib_totale'."""
    top = contribs.head(k).copy()
    return top[["contrib_totale"]]

# -------------------- Bipartitions (taille minimale optionnelle) --------------------

def etiquettes_depuis_indices(n_docs: int, indices_classe1: list) -> list:
    """Construire un vecteur d'étiquettes 0/1 de longueur n_docs, 0 pour classe 1, 1 pour classe 2."""
    lab = [1] * n_docs
    for i in indices_classe1:
        lab[i] = 0
    return lab

def enumerer_bipartitions_filtrees(n_docs: int, taille_min: int) -> list:
    """
    Énumérer les bipartitions en forçant D1 (indice 0) dans la Classe_1 pour éviter les doublons,
    puis filtrer selon la taille minimale par groupe (taille_min).
    Si taille_min == 1, cela équivaut à ne pas imposer de contrainte (autorise 1 vs N−1).
    """
    if n_docs < 2:
        return []
    autres = list(range(1, n_docs))
    partitions = []
    for k in range(0, len(autres)+1):
        for comb in itertools.combinations(autres, k):
            classe1 = [0] + list(comb)
            if len(classe1) == n_docs:
                continue
            classe2 = sorted(set(range(n_docs)) - set(classe1))
            if len(classe1) >= taille_min and len(classe2) >= taille_min:
                partitions.append({"classe1": sorted(classe1), "classe2": classe2})
    return partitions

def extraire_sous_matrice_groupe(tdm_binaire_docs: pd.DataFrame, indices_docs: list) -> pd.DataFrame:
    """Sous-matrice présence/absence pour un groupe donné."""
    noms = [f"D{i+1}" for i in indices_docs]
    return tdm_binaire_docs.loc[noms, :]

def _label_groupe(indices_docs: list, numero: int) -> str:
    """Construire un libellé : 'GROUPE 1 (composé des docs 1, 3, 5)'."""
    nums = ", ".join(str(i+1) for i in indices_docs)
    return f"GROUPE {numero} (composé des docs {nums})"

def construire_objets_pour_bipartitions(corpus: list,
                                        tdm_binaire_docs: pd.DataFrame,
                                        tdm_termes_lignes: pd.DataFrame,
                                        taille_min: int):
    """Assembler, pour chaque bipartition filtrée, textes, matrices et χ², avec libellés 'GROUPE ...'."""
    n_docs = tdm_binaire_docs.shape[0]
    parts = enumerer_bipartitions_filtrees(n_docs, taille_min=taille_min)
    sorties = []
    for p in parts:
        m1 = extraire_sous_matrice_groupe(tdm_binaire_docs, p["classe1"])
        m2 = extraire_sous_matrice_groupe(tdm_binaire_docs, p["classe2"])

        label_g1 = _label_groupe(p["classe1"], 1)
        label_g2 = _label_groupe(p["classe2"], 2)

        etiquettes = etiquettes_depuis_indices(n_docs, p["classe1"])
        table_Vx2 = regrouper_par_classe(tdm_termes_lignes, etiquettes, noms_classes=("Classe_1","Classe_2"))
        table_Vx2 = table_Vx2.rename(columns={"Classe_1": label_g1, "Classe_2": label_g2})
        table_2xV = fabriquer_groupes_en_lignes(table_Vx2)
        chi2 = calculer_chi2_global(table_Vx2)

        textes_c1 = [f"D{i+1} : {corpus[i]}" for i in p["classe1"]]
        textes_c2 = [f"D{i+1} : {corpus[i]}" for i in p["classe2"]]

        sorties.append({
            "classe1_idx": p["classe1"],
            "classe2_idx": p["classe2"],
            "label_g1": label_g1,
            "label_g2": label_g2,
            "classe1_docs": [f"D{i+1}" for i in p["classe1"]],
            "classe2_docs": [f"D{i+1}" for i in p["classe2"]],
            "textes_classe1": textes_c1,
            "textes_classe2": textes_c2,
            "matrice_groupe1_docs": m1,
            "matrice_groupe2_docs": m2,
            "table_regroupée_Vx2": table_Vx2,
            "table_groupes_en_lignes_2xV": table_2xV,
            "chi2": chi2
        })
    return sorties

# -------------------- Interface Streamlit --------------------
st.title("Comprendre la CHD - test du χ²")
st.markdown("""
L’algorithme de Max Reinert vise à découper, grouper et classer un texte. Pour cela, il procède par étapes descendantes : il part du texte et le scinde d’abord en deux groupes, puis continue à diviser chaque corpus de ces deux groupes en sous-groupes. C’est pour cela que l’on parle de classification hiérarchique descendante.

Le principe de calcul est basé sur le test du χ² qui, ici, sert à mesurer la distance entre les groupes/classes.

L’interface Streamlit ci-dessous montre la première division du corpus ; l’algorithme renouvelle ensuite l’opération sur chacun des deux groupes caractérisés par leurs segments de texte.
""")

st.subheader("Étape clé : Matrice termes / documents")
st.markdown("""
1er étape : construire la Matrice Terme-Document binaire.
La matrice binaire indique la présence ou l’absence d’un terme dans un document.

2eme étape : construire la Matrice Termes-Documents REGROUPEE 
La matrice compte le nombre d’occurrences de chaque terme dans chaque document.
Matrice termes-documents-regroupée : Cette matrice est obtenue en faisant la somme de chaque matrice termes-documents, pour chacun des deux groupes de documents. 
Elle résume ainsi la fréquence des termes dans chaque groupe.
C’est cette table « termes × groupes » qui sert au calcul du χ², utilisé ici comme mesure de distance entre les groupes.
""")

# fichier local dans votre repo/projet :
st.image("chd_rainette.png", caption="Illustration de la CHD", use_container_width=True)

texte_defaut = """Le surtourisme crée du stress chez les résidents et affecte leur santé mentale.
Dans certaines villes touristiques, le tourisme de masse accroît le stress et l’anxiété des travailleurs.
Le bruit constant lié au tourisme perturbe le sommeil des résidents et pèse sur la santé mentale.
Les astronomes observent des exoplanètes avec un télescope spatial pour comprendre la formation des systèmes stellaires.
Une supernova illumine une galaxie lointaine et révèle la composition des étoiles.
Les problèmes de santé mentale touchent de plus en plus souvent les étudiants.
La france est le plus grand pays du monde."""

st.subheader("Saisir le corpus : une phrase par ligne")
saisie = st.text_area("Chaque ligne forme un document (D1, D2, ...).", value=texte_defaut, height=260)

st.subheader("Option de contrainte sur la taille des groupes")
activer_taille_min = st.checkbox("Imposer une taille minimale par groupe", value=True)
if activer_taille_min:
    taille_min_ui = st.number_input("Taille minimale par groupe", min_value=1, step=1, value=2)
    st.caption("Quand la contrainte est active, seules les bipartitions où chaque classe contient au moins ce nombre de documents sont conservées.")
else:
    taille_min_ui = 1
    st.caption("La contrainte est désactivée. Toutes les bipartitions sont évaluées, y compris 1 contre N−1.")

corpus = [l.strip() for l in saisie.splitlines() if l.strip()]
n_docs = len(corpus)
if n_docs < 2:
    st.error("Veuillez saisir au moins deux documents.")
    st.stop()

# 1) Matrice binaire (affichée) — filtrage NLTK
tdm_binaire = construire_tdm_binaire_docs_en_lignes(corpus, supprimer_mots_vides=True)
st.subheader("Matrice binaire : présence d'un mot 1/0")
st.dataframe(tdm_binaire, use_container_width=True)
csv_bin = tdm_binaire.to_csv(index=True).encode("utf-8")
st.download_button("Télécharger la matrice binaire (CSV)", csv_bin, "tdm_binaire_docs_lignes.csv", "text/csv")

# 2) Matrice TDM classique (fréquences), harmonisée automatiquement (même vocabulaire que la binaire)
tdm_classique_affichage = construire_tdm(corpus, supprimer_mots_vides=True)
tdm_classique_inversee = tdm_classique_affichage.T  # documents en lignes, termes en colonnes
st.subheader("Matrice Termes-Documents (TDM)")
st.dataframe(tdm_classique_inversee, use_container_width=True)
csv_cls = tdm_classique_inversee.to_csv(index=True).encode("utf-8")
st.download_button("Télécharger la matrice classique (CSV)", csv_cls, "tdm_classique_docs_lignes.csv", "text/csv")
st.caption("Ici, les valeurs sont le cumul des fréquences de mots par document (même vocabulaire que la matrice binaire).")

# 3) Matrice classique POUR LE CALCUL (avec filtrage NLTK) utilisée pour les regroupements et le χ²
tdm_classique_calc = construire_tdm(corpus, supprimer_mots_vides=True)

# 4) Génération des bipartitions selon l'option de taille minimale
sorties = construire_objets_pour_bipartitions(corpus, tdm_binaire, tdm_classique_calc, taille_min=taille_min_ui)

# Récapitulatif χ²
resume = []
for i, s in enumerate(sorties, 1):
    comb1 = ", ".join([f"D{j+1}" for j in s["classe1_idx"]])
    comb2 = ", ".join([f"D{j+1}" for j in s["classe2_idx"]])
    resume.append({
        "id": i,
        "Groupe 1": comb1,
        "Groupe 2": comb2,
        "χ²": round(s["chi2"], 6)
    })
df_resume = pd.DataFrame(resume).set_index("id")
st.subheader("Bipartitions et leur χ²")
st.dataframe(df_resume, use_container_width=True)
csv_res = df_resume.to_csv(index=True).encode("utf-8")
st.download_button("Télécharger le récapitulatif (CSV)", csv_res, "bipartitions_chi2.csv", "text/csv")

# Détail des combinaisons retenues
st.subheader("Détail de toutes les combinaisons retenues")
for i, s in enumerate(sorties, 1):
    st.markdown(f"**Bipartition {i}** — {s['label_g1']} ; {s['label_g2']}")
    st.write(f"χ² global de cette bipartition : {s['chi2']:.6f}")

    st.write("Textes du GROUPE 1")
    for ligne in s["textes_classe1"]:
        st.write(ligne)

    st.write("Textes du GROUPE 2")
    for ligne in s["textes_classe2"]:
        st.write(ligne)

    st.write("Matrice REGROUPÉE (groupes en lignes, termes en colonnes)")
    st.dataframe(s["table_groupes_en_lignes_2xV"], use_container_width=True)
    csv_i = s["table_groupes_en_lignes_2xV"].to_csv(index=True).encode("utf-8")
    st.download_button(f"Télécharger matrice REGROUPÉE bipartition {i} (CSV)", csv_i,
                       f"matrice_regroupee_bipartition_{i}.csv", "text/csv")

    with st.expander("Voir les contributions au χ²"):
        contribs = contributions_par_terme(s["table_regroupée_Vx2"])
        top_contribs = tableau_top_contribs_sans_ecart(contribs, k=K_TOP)
        st.dataframe(top_contribs, use_container_width=True)

# Conclusion : meilleure répartition
if len(sorties) == 0:
    st.warning("Aucune bipartition ne satisfait les paramètres actuels. Ajustez la contrainte de taille minimale ou ajoutez des documents.")
else:
    meilleure = max(sorties, key=lambda x: x["chi2"])
    st.subheader("Meilleure répartition selon le χ²")
    st.write(f"{meilleure['label_g1']} ; {meilleure['label_g2']} ; χ² = {meilleure['chi2']:.6f}")

    col1, col2 = st.columns(2)
    couleur_c1 = "#FFB74D"   # orange
    couleur_c2 = "#64B5F6"   # bleu

    def bloc_phrase_coloree(texte: str, couleur_hex: str) -> None:
        html = f'<div style="background-color:{couleur_hex}; color:#000; padding:10px; border-radius:8px; margin-bottom:8px;">{texte}</div>'
        st.markdown(html, unsafe_allow_html=True)

    with col1:
        st.write("Phrases du GROUPE 1")
        for idx in meilleure["classe1_idx"]:
            bloc_phrase_coloree(f"D{idx+1} : {corpus[idx]}", couleur_c1)

    with col2:
        st.write("Phrases du GROUPE 2")
        for idx in meilleure["classe2_idx"]:
            bloc_phrase_coloree(f"D{idx+1} : {corpus[idx]}", couleur_c2)

    st.write("Matrice REGROUPÉE du meilleur regroupement")
    st.dataframe(meilleure["table_groupes_en_lignes_2xV"], use_container_width=True)
    csv_best = meilleure["table_groupes_en_lignes_2xV"].to_csv(index=True).encode("utf-8")
    st.download_button("Télécharger la matrice REGROUPÉE du meilleur regroupement (CSV)",
                       csv_best, "matrice_regroupee_meilleur.csv", "text/csv")

    with st.expander("Contributions au χ² pour le meilleur regroupement"):
        contribs_best = contributions_par_terme(meilleure["table_regroupée_Vx2"])
        top_contribs_best = tableau_top_contribs_sans_ecart(contribs_best, k=K_TOP)
        st.dataframe(top_contribs_best, use_container_width=True)
