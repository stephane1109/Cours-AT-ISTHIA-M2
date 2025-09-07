# -*- coding: utf-8 -*-

import re

def supprimer_speciaux_sauf_ponctuation(texte):
    """supprime les caractères spéciaux en conservant lettres, chiffres, espaces et ponctuation courante"""
    # on autorise : . , ; : ! ? ' " « » ( ) - – — …
    motif = r"[^0-9A-Za-zÀ-ÖØ-öø-ÿ\s\.\,\;\:\!\?\'\"\«\»\(\)\-\–\—\…]"
    texte = re.sub(motif, " ", texte)
    # normaliser les espaces
    return re.sub(r"\s+", " ", texte).strip()

# exécution directe
chaine = input("Entrez un texte : ")
print(supprimer_speciaux_sauf_ponctuation(chaine))
