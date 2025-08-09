# pip install streamlit matplotlib pandas

import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import io
import re

# Nettoyage et tokenisation
def nettoyer_et_tokeniser(texte):
    texte = texte.lower()
    texte = re.sub(r'[^\w\s]', '', texte)
    mots = texte.split()
    return mots

# Création de la matrice rang / fréquence
def analyser_zipf(mots):
    compteur = Counter(mots)
    data = pd.DataFrame(compteur.items(), columns=['mot', 'frequence'])
    data = data.sort_values(by='frequence', ascending=False).reset_index(drop=True)
    data['rang'] = data.index + 1
    return data

# Affichage de la courbe log-log
def tracer_zipf(dataframes, labels):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dataframes, labels):
        plt.loglog(df['rang'], df['frequence'], label=label)
    plt.xlabel("Rang (échelle logarithmique)")
    plt.ylabel("Fréquence (échelle logarithmique)")
    plt.title("Loi de Zipf - Graphique log-log")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    fig = plt.gcf()
    return fig

# Interface principale
st.title("Analyse textuelle – Loi de Zipf")

fichiers = st.file_uploader("Importer un ou plusieurs fichiers texte (.txt)", type="txt", accept_multiple_files=True)

if fichiers:
    st.divider()
    st.subheader("Résultats pour chaque fichier")

    dataframes = []
    noms_fichiers = []

    for fichier in fichiers:
        texte = fichier.read().decode('utf-8')
        mots = nettoyer_et_tokeniser(texte)
        df = analyser_zipf(mots)
        dataframes.append(df)
        noms_fichiers.append(fichier.name)

        st.markdown(f"### Fichier : `{fichier.name}`")
        st.write(f"- Nombre total de mots : {len(mots)}")
        st.write(f"- Nombre de mots uniques : {len(df)}")

        st.dataframe(df.head(20), use_container_width=True)

        # Export CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Télécharger la matrice CSV",
            data=csv_buffer.getvalue(),
            file_name=f"{fichier.name}_zipf.csv",
            mime="text/csv"
        )

    st.divider()
    st.subheader("Graphique log-log de la loi de Zipf")

    fig = tracer_zipf(dataframes, noms_fichiers)
    st.pyplot(fig)

    # Export PNG
    image_buffer = io.BytesIO()
    fig.savefig(image_buffer, format='png')
    st.download_button(
        label="📸 Télécharger le graphique",
        data=image_buffer.getvalue(),
        file_name="graphique_zipf.png",
        mime="image/png"
    )

    st.divider()
    st.subheader("Interprétation")

    st.markdown("""
La **loi de Zipf** est une observation statistique sur les mots d’un texte :  
Certains mots apparaissent très souvent, d’autres très rarement.

Dans le **graphique log-log** ci-dessus :

- L’**axe horizontal** représente le **rang** du mot : 1er mot le plus fréquent, 2e, 3e, etc.
- L’**axe vertical** représente le **nombre de fois que le mot apparaît**.

Les deux axes sont en **échelle logarithmique**.  
Cela signifie qu’au lieu d’augmenter régulièrement (1, 2, 3, 4...), ils progressent par puissances de 10 :  
**1 → 10 → 100 → 1000**, etc.

Pourquoi faire cela ?  
Parce que cela permet de mieux voir les différences **dans un très grand intervalle de valeurs**.  
Si tu affiches un mot très fréquent (comme "le", "de", "et") avec un mot très rare (comme "capitulaire", "mégajoule"),  
l’échelle normale serait illisible. Le **logarithme compresse visuellement les données** pour que la distribution soit observable.

Enfin, ce que tu vois :  
- Une **chute rapide** au début : ce sont les mots fréquents.
- Une **zone descendante presque droite** au milieu : signe que la loi de Zipf est respectée.
- Une **courbe plate** à la fin : ce sont les mots très rares (Hapax).

---

### Pourquoi c’est important en NLP ?

- Cela montre que **seulement quelques mots dominent le texte**.
- Cela aide à comprendre que **supprimer ou pondérer ces mots** peut améliorer la qualité de la représentation.
- C’est aussi utile pour **trier les mots rares**, souvent synonymes d’information spécifique.
- C’est aussi utile pour **trier les mots trop fréquents**, souvent synonymes de bruit.
- En résumé, **la loi de Zipf aide à mieux modéliser le langage naturel**.

    """)
