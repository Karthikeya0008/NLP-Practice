# Date: 15-08-2026
# Topic: POS Tagging and Chunking
# POS tagging = prathi word ki grammar tag (noun, verb, adjective...), BASICALLY GIVES TAGS!
# Chunking = aa tags use chesi phrases (noun phrase, verb phrase) group cheyyadam

import nltk
from nltk import pos_tag, word_tokenize, RegexpParser

sentence = "The quick brown fox quickly jumped over the lazy dog near the old bridge."
tokens = word_tokenize(sentence)
tags = pos_tag(tokens)
print("POS tags:")

for word, tag in tags:
    print(f"  {word:10s} {tag}")
# DT = determiner, JJ = adjective, NN = noun, VBD = past tense verb, RB = adverb, IN = preposition

# nltk.help.upenn_tagset("VBD")

#Ambiguity - Does my girlfriend like me/ love me ? -- lot of ambiguity!
s1 = "I will book a ticket."
s2 = "I am reading a book."
print("\nAmbiguity check:")
print(" ", pos_tag(word_tokenize(s1)))
print(" ", pos_tag(word_tokenize(s2)))
# s1 lo "book" = VB (verb), s2 lo "book" = NN (noun). context batti marchindi



from collections import Counter
print("\nTag counts:", Counter(tag for _, tag in tags).most_common())

grammar = r"NP: {<DT>?<JJ>*<NN.*>+}"
parser = RegexpParser(grammar)
tree = parser.parse(tags)
print("\nChunk tree:")
print(tree)

print("\nNoun phrases found:")
for subtree in tree.subtrees(filter=lambda t: t.label() == "NP"):
    print("  ", " ".join(word for word, _ in subtree.leaves()))

grammar2 = r"""
  NP: {<DT>?<JJ>*<NN.*>+}
  PP: {<IN><NP>}
  VP: {<RB>?<VB.*><PP|NP>*}
"""
print("\nNP + PP + VP chunks:")
print(RegexpParser(grammar2).parse(tags))


messi = "The legendary Argentine captain Lionel Messi scored a stunning free kick against Brazil."
m_tags = pos_tag(word_tokenize(messi))
print("\nMessi POS tags:", m_tags)
print("Messi noun phrases:")
for st in parser.parse(m_tags).subtrees(filter=lambda t: t.label() == "NP"):
    print("  ", " ".join(w for w, _ in st.leaves()))
# "Lionel Messi" NNP NNP -> Proper Noun, justified as Messi is a name.
# "free kick" -> "free" got JJ tag.

chink_grammar = r"""
  Chunk: {<.*>+}
         }<VB.*|IN>{
"""
print("\nChinking (verbs and prepositions removed from chunks):")
print(RegexpParser(chink_grammar).parse(tags))
