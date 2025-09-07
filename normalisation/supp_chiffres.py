import re

def nettoyer_texte(texte):
    """supprime tous les chiffres du texte."""
    return re.sub(r"\d+", " ", texte)

# exécution
texte_utilisateur = input("Entrez une phrase contenant des chiffres : ")
resultat = nettoyer_texte(texte_utilisateur)
print("\nRésultat :")
print(resultat)
