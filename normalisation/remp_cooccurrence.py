# remplacement des termes par : terme1_terme2
# texte :
# Paris Match publie un dossier sur Steve Jobs et son impact sur l’industrie.
# L’article met aussi en regard les promesses de l’intelligence artificielle.
# Certains lecteurs comparent Paris match d’hier et d’aujourd’hui, tandis que d’autres évoquent l’Intelligence Artificielle dans les produits d’Apple inspirés par Steve Jobs.

import re

def remplacer_occurrences_multiples(texte, remplacements):
    """remplace toutes les occurrences par leur valeur correspondante"""
    for a_chercher, remplacement in remplacements.items():
        motif = re.escape(a_chercher)  # traiter la clé comme texte littéral
        texte = re.sub(motif, remplacement, texte, flags=re.IGNORECASE) # on ignore la CASE
    return texte

# saisir le texte à transformer
texte_source = input("Entrez le texte à transformer : ")

# DÉFINIR LES RÈGLES DE REMPLACEMENT
# exemple : "Paris Match" -> "Paris_Match"
remplacements = {
    "Paris Match": "Paris_Match",
    "Steve Jobs": "Steve_Jobs",
    "intelligence artificielle": "intelligence_artificielle"
    # ajoutez d'autres lignes si nécessaire :
}

# appliquer les remplacements et afficher le résultat
resultat = remplacer_occurrences_multiples(texte_source, remplacements)
print("\nTexte transformé :")
print(resultat)