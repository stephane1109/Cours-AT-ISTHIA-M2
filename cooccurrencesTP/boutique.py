# python -m streamlit run main.py
# pip install wordcloud matplotlib pandas networkx streamlit

# Analyse des cooccurrences d’articles achetés à partir d’un CSV :
# - une ligne = un acheteur
# - une colonne (ex. "Achats") = liste d'articles séparés par des virgules
# Calcul et visualisation sans bouton : le graphe se met à jour lorsque les paramètres changent.

import streamlit as st
import pandas as pd
import itertools
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

# -------------------------------------------------------------------
# Fonctions utilitaires
# -------------------------------------------------------------------

def parser_ligne_achats(texte):
    """Découpe une chaîne d'achats en liste d'articles, en nettoyant espaces et entrées vides."""
    if pd.isna(texte):
        return []
    items = [a.strip() for a in str(texte).split(",")]
    return [a for a in items if a]

@st.cache_data(show_spinner=False)
def calculer_cooccurrences(df, nom_colonne_achats):
    """
    Calcule toutes les cooccurrences d’articles sur l’ensemble des paniers.
    Retourne un DataFrame trié par 'frequence' décroissante et la liste triée des articles uniques.
    """
    cooc = Counter()
    vocab = set()

    for achats in df[nom_colonne_achats]:
        items = parser_ligne_achats(achats)
        if not items:
            continue
        items_unique = sorted(set(items))  # déduplication par panier
        vocab.update(items_unique)
        for a1, a2 in itertools.combinations(items_unique, 2):
            cooc[(a1, a2)] += 1

    data = [{"article_1": a1, "article_2": a2, "frequence": freq}
            for (a1, a2), freq in cooc.items()]
    df_cooc = pd.DataFrame(data, columns=["article_1", "article_2", "frequence"])
    if not df_cooc.empty:
        df_cooc = df_cooc.sort_values(by="frequence", ascending=False).reset_index(drop=True)

    return df_cooc, sorted(vocab)

def tracer_graphe_cooccurrences(df_cooc, mode_filtrage="top", top_n=20, seuil=1, taille_fig=(14, 10), espacement=0.6):
    """
    Trace un graphe non orienté des cooccurrences avec options de taille et d'espacement.
    Sauvegarde le graphe en PNG et propose un téléchargement.
    """
    if df_cooc.empty:
        st.info("La matrice de cooccurrences est vide.")
        return

    if mode_filtrage == "top":
        subset = df_cooc.head(top_n)
        titre = f"Graphe des {min(top_n, len(subset))} paires les plus fréquentes"
    else:
        subset = df_cooc[df_cooc["frequence"] >= seuil].copy()
        subset = subset.sort_values("frequence", ascending=False)
        titre = f"Graphe des paires avec fréquence ≥ {seuil} (total {len(subset)})"

    if subset.empty:
        st.info("Aucune paire à afficher avec ce filtrage.")
        return

    G = nx.Graph()
    for _, row in subset.iterrows():
        a1, a2, w = row["article_1"], row["article_2"], int(row["frequence"])
        G.add_edge(a1, a2, weight=w, label=str(w))

    fig, ax = plt.subplots(figsize=taille_fig)
    pos = nx.spring_layout(G, seed=42, k=espacement)

    widths = [max(1.0, float(d["weight"])) for (_, _, d) in G.edges(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=1200, node_color="lightblue", edgecolors="black", linewidths=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)

    edge_labels = {(u, v): d["label"] for (u, v, d) in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.5, ax=ax)

    ax.set_title(titre)
    ax.axis("off")
    st.pyplot(fig)

    # Sauvegarde du graphe en PNG
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    st.download_button(
        label="Télécharger le graphe en PNG",
        data=buffer,
        file_name="graphe_cooccurrences.png",
        mime="image/png"
    )

# -------------------------------------------------------------------
# Interface Streamlit
# -------------------------------------------------------------------

st.title("Analyse des cooccurrences d’articles achetés")

st.markdown(
    "Chargez un CSV contenant une colonne avec la liste des articles achetés par acheteur "
    "(séparés par des virgules). Le script calcule toutes les cooccurrences et affiche un graphe "
    "dont chaque arête porte le nombre de cooccurrences observées."
)

# Téléversement du CSV
fichier = st.file_uploader("Importer le fichier CSV", type=["csv"], key="uploader_csv_achats")

if fichier is not None:
    try:
        df = pd.read_csv(fichier)
    except Exception as e:
        st.error(f"Erreur de lecture du CSV : {e}")
        st.stop()

    if df.empty:
        st.error("Le CSV chargé est vide.")
        st.stop()

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    colonnes_candidats = [c for c in df.columns if c.lower() != "acheteur"]
    if not colonnes_candidats:
        st.error("Aucune colonne candidate trouvée (une colonne texte 'Achats' est attendue).")
        st.stop()

    nom_colonne_achats = st.selectbox("Sélectionnez la colonne contenant les listes d’articles :", colonnes_candidats)

    df_cooc, vocab = calculer_cooccurrences(df, nom_colonne_achats)

    st.subheader("Tableau des cooccurrences")
    if df_cooc.empty:
        st.info("Aucune cooccurrence détectée. Vérifiez le format de la colonne choisie.")
    else:
        st.dataframe(df_cooc)

        csv_cooc = df_cooc.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger le tableau des cooccurrences (CSV)",
            data=csv_cooc,
            file_name="cooccurrences_articles.csv",
            mime="text/csv"
        )

        st.subheader("Paires les plus fréquentes")
        top5 = df_cooc.head(5)
        st.dataframe(top5)
        if not top5.empty:
            best = df_cooc.iloc[0]
            st.success(
                f"Le panier le plus fréquent contient : « {best['article_1']} » et « {best['article_2']} », "
                f"achetés ensemble {int(best['frequence'])} fois."
            )

        # Paramètres visuels
        st.subheader("Paramètres du graphe")
        taille_x = st.slider("Largeur du graphe (pouces)", 8, 30, 14)
        taille_y = st.slider("Hauteur du graphe (pouces)", 6, 20, 10)
        espacement = st.slider("Espacement entre les nœuds (k)", 0.1, 2.0, 0.6)

        mode = st.radio("Mode de filtrage :", ["Top N paires", "Seuil de fréquence"], index=0)
        if mode == "Top N paires":
            top_n_graph = st.slider("Nombre de paires à afficher", 5, 200, 20, step=5)
            tracer_graphe_cooccurrences(df_cooc, mode_filtrage="top", top_n=top_n_graph,
                                        taille_fig=(taille_x, taille_y), espacement=espacement)
        else:
            seuil = st.slider("Afficher les paires avec fréquence ≥ ", 1, 100, 5, step=1)
            tracer_graphe_cooccurrences(df_cooc, mode_filtrage="seuil", seuil=seuil,
                                        taille_fig=(taille_x, taille_y), espacement=espacement)
else:
    st.info(
        "Importez un CSV pour démarrer.\n"
        "Format attendu : une colonne texte avec des articles séparés par des virgules.\n"
        "Exemple : Acheteur_01,\"Tasse Château de Foix, Set de table Château de Foix, Affiche n°2\""
    )
