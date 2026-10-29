# Date: 01-09-2026
# Topic: WordNet
# WordNet = English ki peddha dictionary + thesaurus. words ni "synsets" (same meaning groups) ga store chestundi
# synonyms, antonyms, definitions, hypernyms (parent concept) and etc are present!.

from nltk.corpus import wordnet as wn

word = "bank"
syns = wn.synsets(word)
print(f"'{word}' ki {len(syns)} meanings unnayi. modati 5:")
for s in syns[:5]:
    print(f"  {s.name():25s} {s.definition()}")


# Definition and examplees:
car = wn.synset("car.n.01")
print("\ncar.n.01 definition:", car.definition())
print("examples:", car.examples())
print("lemma names:", car.lemma_names())   # same synset lo unna synonyms

#Synonyyms and Antonyms:
def syn_ant(word):
    syn, ant = set(), set()
    for s in wn.synsets(word):
        for lemma in s.lemmas():
            syn.add(lemma.name())
            for a in lemma.antonyms():
                ant.add(a.name())
    return syn, ant

for w in ["good", "happy", "fast"]:
    s, a = syn_ant(w)
    print(f"\n{w}: synonyms = {sorted(s)[:8]}")
    print(f"{w}: antonyms = {sorted(a)}")
# antonyms lemma level lo untayi, synset level lo kaadu. 
#Anotonyms -> Lemma Level!

dog = wn.synset("dog.n.01")
print("\ndog hypernyms (parent):", dog.hypernyms())
print("dog hyponyms (children, first 5):", dog.hyponyms()[:5])

print("\ndog nunchi root varaku path:")
path = dog.hypernym_paths()[0]
print("  " + " -> ".join(s.name().split(".")[0] for s in path))

pairs = [("dog", "cat"), ("dog", "wolf"), ("dog", "car"), ("ship", "boat")]
print("\nWu-Palmer similarity:")
for a, b in pairs:
    s1, s2 = wn.synset(f"{a}.n.01"), wn.synset(f"{b}.n.01")
    print(f"  {a:5s} vs {b:5s} -> {s1.wup_similarity(s2):.3f}")
print("\nPath similarity dog vs cat:", round(wn.synset("dog.n.01").path_similarity(wn.synset("cat.n.01")), 3))

print("\n'goal' meanings:")
for s in wn.synsets("goal", pos=wn.NOUN):
    print(f"  {s.name():15s} {s.definition()}")
# football goal and life goal (aim) both are separate synsets

print("\nfootball synonyms:", wn.synset("soccer.n.01").lemma_names())
# "soccer" and "association_football" are present in the same synset.
print("soccer hypernym path:",
      " -> ".join(s.name().split(".")[0] for s in wn.synset("soccer.n.01").hypernym_paths()[0][-4:]))

dribble = wn.synsets("dribble", pos=wn.VERB)
print("\n'dribble' verb senses:", [s.definition() for s in dribble])
# Messi famous ayina dribble ki "ball ni kick chestu tisukellatam" ane sense kuda undi

print("\ndog in other languages:")
for lang in ["spa", "fra", "jpn"]:
    print(f"  {lang}: {dog.lemma_names(lang)[:3]}")
# Spanish lo "perro" vachindi, matches what i learnt on Duo Lingo!, also the startup i worked for -> MyPerro!
