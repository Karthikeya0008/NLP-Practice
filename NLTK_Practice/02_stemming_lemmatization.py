# Date: 07-08-2026
# Topic: Stemming vs Lemmatization
# Stemming = rules tho word chivara cut cheyyadam (meaning check cheyyadu)
# Lemmatization = dictionary (WordNet), gives the correct word (lemma) based on meaning.

from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

porter = PorterStemmer()
lancaster = LancasterStemmer()
snowball = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

words = ["running", "runs", "ran", "easily", "fairly", "studies", "studying",
         "better", "wolves", "geese", "organization", "university", "happiness"]

print(f"{'word':14s}{'porter':12s}{'lancaster':12s}{'snowball':12s}")
for w in words:
    print(f"{w:14s}{porter.stem(w):12s}{lancaster.stem(w):12s}{snowball.stem(w):12s}")


print("\nLemmatizer (default pos = noun):")
for w in words:
    print(f"  {w:14s} -> {lemmatizer.lemmatize(w)}")

print("\nWith correct POS:")
print("  running (v) ->", lemmatizer.lemmatize("running", pos="v"))
print("  ran     (v) ->", lemmatizer.lemmatize("ran", pos="v"))
print("  better  (a) ->", lemmatizer.lemmatize("better", pos="a"))
# "ran" -> "run" and "better" -> "good" 


sentence = "The children were playing happily while their parents were watching the matches."
stop = set(stopwords.words("english"))
tokens = [t.lower() for t in word_tokenize(sentence) if t.isalpha()]
content = [t for t in tokens if t not in stop]

print("\nAfter stopword removal:", content)
print("Stemmed    :", [porter.stem(t) for t in content])
print("Lemmatized :", [lemmatizer.lemmatize(t) for t in content])


#LEO LEO LEO LEO LEO Messi sentence - THE GOAATTTTTTT!
messi = "Messi was dribbling past defenders and scored goals in matches that fans remembered for years."
m_tokens = [t.lower() for t in word_tokenize(messi) if t.isalpha() and t.lower() not in stop]
print("\nMessi tokens:", m_tokens)
print("Stemmed   :", [porter.stem(t) for t in m_tokens])
print("Lemma (v) :", [lemmatizer.lemmatize(t, pos="v") for t in m_tokens])
# "dribbling" -> stem "dribbl" (word kaadu), lemma with pos v -> "dribble"
# "scored" -> "score" rendu lo vachindi. "messi" ni stemmer "messi" ga ne unchindi

# Summary:
#  stemming fast, but output words may not necessarily be in dictionary.
#  lemmatization slow, kani correct words istundi, POS kavali
#  search engines lo stemming saripotundi, chatbot / NER lanti vaatiki lemmatization better
