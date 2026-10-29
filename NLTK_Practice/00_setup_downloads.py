# Date: 02-08-2026
# NLTK setup - first time run cheyyali

import nltk
packages = [
    "punkt", "punkt_tab",
    "stopwords",
    "wordnet", "omw-1.4",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker", "maxent_ne_chunker_tab", "words",
    "gutenberg", "brown", "movie_reviews", "inaugural",
]

for p in packages:
    ok = nltk.download(p, quiet=True)
    print(f"{p:35s} -> {'done' if ok else 'FAILED'}")

print("\nAnni download ayyayi, ippudu practice start cheyyochu")
