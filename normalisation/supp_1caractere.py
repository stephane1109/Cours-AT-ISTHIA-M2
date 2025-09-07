# Nettoyage de caractères isolés

import re

def supprimer_caracteres_isoles(texte):
    """supprime les mots d'une seule lettre tout en conservant les formes avec apostrophe (c', j', l', etc.)"""
    # normaliser l’apostrophe typographique en apostrophe droite
    texte = texte.replace("’", "'")
    # retirer une lettre isolée (pas précédée/suivie d'un caractère de mot) et non suivie d'une apostrophe
    texte = re.sub(r"(?<!\w)[A-Za-zÀ-ÖØ-öø-ÿ](?!\w|')", " ", texte)
    # nettoyer les espaces multiples
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte

# exécution
phrase = input("Entrez une phrase : ")
resultat = supprimer_caracteres_isoles(phrase)
print("\nRésultat :")
print(resultat)