# python -m streamlit run main.py
# streamlit run main.py
# pip install wordcloud matplotlib

import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import re
from io import BytesIO


# Fonction de nettoyage et tokenisation
def nettoyer_et_tokeniser(texte):
    texte = texte.lower()
    texte = re.sub(r"[^\w\s]", "", texte)
    mots = texte.split()
    return mots

# Générer un nuage de mots
def generer_nuage_de_mots(freq, titre):
    nuage = WordCloud(
        width=800,
        height=400,
        background_color='white',
        colormap='viridis'
    ).generate_from_frequencies(freq)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(nuage, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(titre)
    return fig

# Interface utilisateur
st.title("Analyse de fréquence et nuage de mots")
st.markdown("Importez un ou plusieurs fichiers `.txt`. Visualisez les **mots les plus fréquents** sous forme de nuage, et **téléchargez la matrice** `mot / fréquence / rang`.")

fichiers = st.file_uploader("Importer vos fichiers texte", type="txt", accept_multiple_files=True)

if fichiers:
    st.sidebar.header("Paramètres")
    nb_mots = st.sidebar.slider("Nombre de mots dans le nuage", 10, 100, 20, step=5)

    for fichier in fichiers:
        texte = fichier.read().decode("utf-8")
        mots = nettoyer_et_tokeniser(texte)
        freqs = Counter(mots)
        mots_cles = dict(freqs.most_common(nb_mots))

        st.markdown(f"## {fichier.name}")
        st.write(f"Nombre total de mots : {len(mots)}")
        st.write(f"Nombre de mots uniques : {len(set(mots))}")

        # Affichage du nuage de mots
        fig = generer_nuage_de_mots(mots_cles, f"Nuage de mots – {fichier.name}")
        st.pyplot(fig)

        # Export PNG du nuage de mots
        buffer_image = BytesIO()
        fig.savefig(buffer_image, format="png")
        buffer_image.seek(0)

        st.download_button(
            label="Télécharger le nuage (PNG)",
            data=buffer_image,
            file_name=f"{fichier.name}_nuage.png",
            mime="image/png",
            key=f"{fichier.name}_png"  # clé unique par fichier
        )

        # Affichage du graphique en barres des mots les plus fréquents
        st.subheader(f"Graphique en barres des {nb_mots} mots les plus fréquents")

        fig_bar, ax = plt.subplots(figsize=(12, 6))  # taille agrandie
        mots_selectionnes = list(mots_cles.keys())
        frequences_selectionnees = list(mots_cles.values())

        ax.bar(mots_selectionnes, frequences_selectionnees, color='skyblue')
        ax.set_xlabel("Mots", fontsize=12)
        ax.set_ylabel("Fréquence", fontsize=12)
        ax.set_title(f"Top {nb_mots} mots les plus fréquents", fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        st.pyplot(fig_bar)

        # Export PNG du graphique en barres
        buffer_bar = BytesIO()
        fig_bar.savefig(buffer_bar, format="png")
        buffer_bar.seek(0)

        st.download_button(
            label="Télécharger le graphique en barres (PNG)",
            data=buffer_bar,
            file_name=f"{fichier.name}_barres.png",
            mime="image/png",
            key=f"{fichier.name}_barres_png"
        )

        # Création de la matrice fréquence/rang
        donnees = [(mot, freq) for mot, freq in freqs.items()]
        donnees_tries = sorted(donnees, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(donnees_tries, columns=["mot", "frequence"])
        df["rang"] = range(1, len(df) + 1)

        # Affichage tableau
        st.subheader("Matrice `mot / fréquence / rang`")
        st.dataframe(df.head(50))

        # Export CSV (corrigé avec key unique)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Télécharger la matrice CSV",
            data=csv,
            file_name=f"{fichier.name}_frequences.csv",
            mime="text/csv",
            key=f"{fichier.name}_csv"  # clé unique par fichier
        )

    st.divider()
    st.subheader("Explication : intérêt et limites")
    st.markdown(f"""
### Intérêt
- Le **nuage de mots** permet une visualisation synthétique des termes frequents.
- Le tableau `mot / fréquence / rang` offre une base exploitable pour l'analyse statistique.

### Limites
- Les **fréquences ne rendent pas compte du sens** ni du contexte.
- **Aucune relation grammaticale** ou sémantique n’est représentée.
- Ce prétraitement reste **brut** (pas de lemmatisation, ni de filtrage grammatical).

Pour une analyse plus avancée : NLP, TF-IDF, cooccurrences, réseaux lexicaux...
""")
