# fonction de nettoyage : mettre le texte en minuscules
def nettoyer_texte(texte, conserver_casse=True):
    """retourne le texte en minuscules si conserver_casse est False, sinon tel quel."""
    if not conserver_casse:
        texte = texte.lower()
    return texte

# exécution
texte_utilisateur = input("Entrez un texte à tester : ")
resultat = nettoyer_texte(texte_utilisateur, conserver_casse=False)
print("\nRésultat :")
print(resultat)