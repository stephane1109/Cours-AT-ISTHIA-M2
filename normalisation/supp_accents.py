# normalisation des mots avec accents

import unicodedata

def supprimer_accents(texte):
    """supprime accents et ligatures (œ→oe, æ→ae) en conservant casse et ponctuation"""
    # gérer explicitement les ligatures courantes
    texte = (texte
             .replace("œ", "oe").replace("Œ", "OE")
             .replace("æ", "ae").replace("Æ", "AE"))
    # normaliser puis retirer tous les diacritiques (catégories Unicode M*)
    decomp = unicodedata.normalize("NFKD", texte)
    return "".join(ch for ch in decomp if not unicodedata.category(ch).startswith("M"))

# exécution
phrase = input("Entrez une phrase avec accents et ligatures : ")
resultat = supprimer_accents(phrase)
print("\nRésultat :")
print(resultat)

# Exemple à tester :
# À Â Ä Ç É È Ê Ë Î Ï Ô Ö Ù Û Ü Ÿ — à â ä ç é è ê ë î ï ô ö ù û ü ÿ — œ, Œ, æ, Æ.
