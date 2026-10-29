# Date: 27-08-2026
# Topic: N-grams
# N-gram = N words that come in a sequence. unigram (1), bigram (2), trigram (3)
# language model: to predict the next word... base of all models!, prediction of which word comes next!...

from nltk import ngrams, bigrams, trigrams, word_tokenize, FreqDist
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.corpus import gutenberg
from collections import defaultdict, Counter

sentence = "I love natural language processing and I love machine learning"
tokens = word_tokenize(sentence.lower())

print("Unigrams:", tokens)
print("Bigrams :", list(bigrams(tokens)))
print("Trigrams:", list(trigrams(tokens)))
print("4-grams :", list(ngrams(tokens, 4))[:3], "...")

words = [w.lower() for w in gutenberg.words("austen-emma.txt") if w.isalpha()]
print("\nEmma novel total words:", len(words))

bigram_freq = FreqDist(bigrams(words))
print("Top 10 bigrams:")
for bg, c in bigram_freq.most_common(10):
    print(f"  {' '.join(bg):20s} {c}")

finder = BigramCollocationFinder.from_words(words)
finder.apply_freq_filter(5)        # 5 sarla kante takkuva vachinavi teesesta
print("\nTop collocations by PMI:")
print(" ", finder.nbest(BigramAssocMeasures.pmi, 10))

model = defaultdict(Counter)
for w1, w2 in bigrams(words):
    model[w1][w2] += 1

def next_word_probs(word, k=5):
    total = sum(model[word].values())
    return [(w, round(c / total, 3)) for w, c in model[word].most_common(k)]

for w in ["mr", "she", "very"]:
    print(f"\nAfter '{w}':", next_word_probs(w))

def generate(start, length=10):
    out = [start]
    for _ in range(length):
        if not model[out[-1]]:
            break
        out.append(model[out[-1]].most_common(1)[0][0])
    return " ".join(out)

print("\nGenerated:", generate("she"))

# LEO MESSI STRIKES AGAIN!  
messi_text = ("Messi scores a goal. Messi scores again. Messi wins the match. "
              "Argentina wins the World Cup. Messi lifts the World Cup.")
m_tokens = [w.lower() for w in word_tokenize(messi_text) if w.isalpha()]
m_bigrams = FreqDist(bigrams(m_tokens))
print("\nMessi bigrams:", m_bigrams.most_common(5))


m_model = defaultdict(Counter)
for a, b in bigrams(m_tokens):
    m_model[a][b] += 1
total = sum(m_model["messi"].values())
print("P(next | messi):", {w: round(c / total, 2) for w, c in m_model["messi"].items()})

# TODO: smoothing (add-one / Laplace):
# to check the unseen probabilities
