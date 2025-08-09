# Scrapinge de wikipedia avec l'API

import requests
import re

# Fonction pour extraire le contenu brut depuis l'API Wikipédia
def extraire_texte_article_wikipedia(titre_article, langue='fr'):
    url_api = f"https://{langue}.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'prop': 'extracts',
        'explaintext': True,
        'exsectionformat': 'plain',
        'titles': titre_article,
        'format': 'json'
    }

    reponse = requests.get(url_api, params=params)
    data = reponse.json()
    page = next(iter(data['query']['pages'].values()))

    if 'extract' in page:
        texte = page['extract']
        return texte
    else:
        raise ValueError("Aucun texte trouvé pour cet article.")

# Fonction de nettoyage pour conserver uniquement les parties utiles
def filtrer_sections_utiles(texte):
    """
    Garde uniquement les sections informatives : introduction, géographie, climat, etc.
    Supprime les sections non pertinentes (devise, bibliographie, voir aussi...).
    """
    # Définir les sections à garder (en minuscules sans accents pour robustesse)
    sections_a_garder = [
        "géographie", "localisation", "géologie et relief", "paysages",
        "hydrographie", "climat"
    ]

    # Supprimer les sections à partir de "Notes et références" et après
    sections_a_supprimer = [
        "notes et références", "notes", "cartes", "voir aussi", "bibliographie",
        "liens externes", "portail", "articles connexes", "devise"
    ]

    # Nettoyer les titres : créer une liste (titre, contenu)
    blocs = re.split(r'\n(?=[A-Z][^\n]{0,80}\n)', texte)
    texte_filtré = blocs[0]  # introduction (toujours présente)

    for bloc in blocs[1:]:
        # Identifier le titre de section
        lignes = bloc.strip().split('\n')
        if not lignes:
            continue
        titre = lignes[0].strip().lower()

        titre = re.sub(r'[^\w\s]', '', titre)  # retirer ponctuation
        titre = re.sub(r'\s+', ' ', titre)

        if any(titre.startswith(s) for s in sections_a_garder) and not any(titre.startswith(s) for s in sections_a_supprimer):
            texte_filtré += '\n\n' + bloc.strip()

    return texte_filtré.strip()

# Fonction d'enregistrement dans un fichier .txt
def enregistrer_texte_nettoye(titre_article, nom_fichier):
    texte_complet = extraire_texte_article_wikipedia(titre_article)
    texte_filtré = filtrer_sections_utiles(texte_complet)
    with open(nom_fichier, 'w', encoding='utf-8') as f:
        f.write(texte_filtré)
    print(f"Fichier enregistré : {nom_fichier}")

# Utilisation
if __name__ == "__main__":
    enregistrer_texte_nettoye("Bordeaux", "article_bordeaux_nettoye.txt")
    enregistrer_texte_nettoye("Paris", "article_paris_nettoye.txt")


