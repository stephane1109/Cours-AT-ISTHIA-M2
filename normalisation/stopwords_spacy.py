# pip install spacy
# python -m spacy download fr_core_news_sm


import spacy

# charger le modèle français de spaCy
# installez-le au besoin : python -m spacy download fr_core_news_sm
nlp = spacy.load("fr_core_news_sm")

def supprimer_mots_vides_spacy(texte):
    """supprime les mots vides français avec spacy et conserve l'ordre des mots"""
    doc = nlp(texte)
    mots = [t.text for t in doc if not t.is_stop and not t.is_punct and not t.is_space]
    return " ".join(mots)

# exécution
phrase = input("Entrez une phrase : ")
print(supprimer_mots_vides_spacy(phrase))

