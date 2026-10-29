# Date: 05-09-2026
# Topic: NLTK built-in corpora
# Corpus = peddha text collection. NLTK lo chala ready-made corpora unnayi. like manyyyy!
# Gutenberg (books), Brown (categories), Inaugural (US speeches), Movie Reviews (sentiment), Stopwords

from nltk.corpus import gutenberg, brown, inaugural, movie_reviews, stopwords
from nltk import FreqDist, ConditionalFreqDist
import random

print("Gutenberg files:", gutenberg.fileids()[:6], "...")
for fid in ["austen-emma.txt", "shakespeare-hamlet.txt", "bible-kjv.txt"]:
    n_chars = len(gutenberg.raw(fid))
    n_words = len(gutenberg.words(fid))
    n_sents = len(gutenberg.sents(fid))
    vocab = len(set(w.lower() for w in gutenberg.words(fid)))
    print(f"  {fid:25s} avg word len={n_chars/n_words:.1f}  "
          f"avg sent len={n_words/n_sents:.1f}  lexical diversity={vocab/n_words:.3f}")

print("\nBrown categories:", brown.categories())

cfd = ConditionalFreqDist(
    (genre, w.lower())
    for genre in ["news", "religion", "romance", "science_fiction"]
    for w in brown.words(categories=genre)
)
modals = ["can", "could", "may", "might", "must", "will"]
print("\nModal verbs by genre:")
cfd.tabulate(conditions=["news", "religion", "romance", "science_fiction"], samples=modals)

print("\nInaugural speeches:", len(inaugural.fileids()), "files")
cfd2 = ConditionalFreqDist(
    (target, fid[:4])
    for fid in inaugural.fileids()
    for w in inaugural.words(fid)
    for target in ["america", "citizen"]
    if w.lower().startswith(target)
)

print("'america' mentions in last 5 speeches:",
      [(y, cfd2["america"][y]) for y in sorted(cfd2["america"])[-5:]])


print("\nMovie review categories:", movie_reviews.categories())
print("pos files:", len(movie_reviews.fileids("pos")), " neg files:", len(movie_reviews.fileids("neg")))

docs = [(list(movie_reviews.words(f)), c) for c in movie_reviews.categories()
        for f in movie_reviews.fileids(c)]
random.seed(42)
random.shuffle(docs)

stop = set(stopwords.words("english"))
all_words = FreqDist(w.lower() for w in movie_reviews.words() if w.isalpha() and w.lower() not in stop)
top_words = [w for w, _ in all_words.most_common(2000)]

def features(doc):
    d = set(doc)
    return {f"has({w})": (w in d) for w in top_words}

import nltk
featuresets = [(features(d), c) for d, c in docs]
train, test = featuresets[200:], featuresets[:200]
clf = nltk.NaiveBayesClassifier.train(train)
print("\nNaive Bayes accuracy:", round(nltk.classify.accuracy(clf, test), 3))
clf.show_most_informative_features(8)

print("\nStopword languages:", len(stopwords.fileids()))
print("Spanish stopwords (first 10):", stopwords.words("spanish")[:10])

my_review = ("Watching Messi play was outstanding. His vision and passing were brilliant, "
             "a wonderful and memorable performance.").lower().split()
print("\nMessi review prediction:", clf.classify(features([w.strip(".,") for w in my_review])))

