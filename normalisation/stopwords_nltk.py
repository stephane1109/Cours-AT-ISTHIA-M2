# pip install nltk

import re

# charger les stopwords français nltk
try:
    from nltk.corpus import stopwords
    stop_fr = set(stopwords.words("french"))
except LookupError:
    import nltk
    nltk.download("stopwords")
    from nltk.corpus import stopwords
    stop_fr = set(stopwords.words("french"))

def supprimer_mots_vides_nltk(texte):
    """supprime les mots vides français avec nltk et conserve l'ordre des mots"""
    mots = re.findall(r"\w+", texte.lower(), flags=re.UNICODE)
    mots_filtres = [m for m in mots if m not in stop_fr]
    return " ".join(mots_filtres)

# exécution
phrase = input("Entrez une phrase : ")
print(supprimer_mots_vides_nltk(phrase))