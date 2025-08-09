import re

# Fonction pour afficher les résultats de la tokenisation
def afficher_resultats(texte, regex, description):
    tokens = re.findall(regex, texte)
    print(f"\n{description}:")
    print(tokens)

# Exemple de texte pour la démonstration
texte1 = "L'adresse d'un utilisateur est : 123@exemple.com."
texte2 = "Les arbres, les fleurs et les oiseaux sont beaux!"
texte3 = "Python est un langage puissant. Il est très utilisé en science des données."

# 1. Utilisation de \w+ (Règle par défaut)
# Capture des mots alphanumériques sans tenir compte de la ponctuation
regex_w_plus = r'\w+'

afficher_resultats(texte1, regex_w_plus, "1. Tokenisation avec \\w+ (Capture des mots alphanumériques)")

# 2. Utilisation de \b\w+\b (Délimitation stricte des mots)
# Délimitation stricte des mots en évitant les symboles comme l'apostrophe ou les signes de ponctuation.
regex_b_w_plus_b = r'\b\w+\b'

afficher_resultats(texte1, regex_b_w_plus_b, "2. Tokenisation avec \\b\\w+\\b (Délimitation stricte des mots)")

# 3. Utilisation de \s+ pour séparer le texte par les espaces
# Capture chaque segment séparé par un espace
regex_s_plus = r'\S+'  # Utilisation de \S+ pour capturer des segments non-espaces

afficher_resultats(texte2, regex_s_plus, "3. Tokenisation avec \\S+ (Séparation par espaces)")

# 4. Utilisation de la tokenisation par phrases (basée sur des délimiteurs de fin de phrase)
# Capture chaque phrase en utilisant les points ou autres délimiteurs de fin de phrase
regex_par_phrase = r'[^.!?]*[.!?]'

afficher_resultats(texte3, regex_par_phrase, "4. Tokenisation par phrase (Séparation par délimiteurs de fin de phrase)")
