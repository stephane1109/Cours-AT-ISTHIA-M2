# suppression des url

import re

def supprimer_http(texte):
    """supprime les expressions qui commencent par http (ou https) et tout ce qui suit jusqu'au prochain espace"""
    # enlever chaque segment 'http...' (jusqu'au premier séparateur d'espace)
    texte = re.sub(r"http\S+", " ", texte)
    # normaliser les espaces
    return re.sub(r"\s+", " ", texte).strip()

# exécution
phrase = input("Entrez une phrase contenant des liens http : ")
resultat = supprimer_http(phrase)
print("\nRésultat :")
print(resultat)
