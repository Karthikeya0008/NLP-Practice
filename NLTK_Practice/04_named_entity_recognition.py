# Date: 21-08-2026
# Topic: Named Entity Recognition (NER)
# NER = recognises names,text,location,organization,relations, etc....
# NLTK lo ne_chunk() function, POS tags meeda base ayi panichestundi

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

text = ("Sundar Pichai, the CEO of Google, visited Hyderabad on Monday. "
        "He met officials from Infosys and the Government of Telangana.")

tree = ne_chunk(pos_tag(word_tokenize(text)))
print(tree)

def get_entities(tree):
    entities = []
    for node in tree:
        if isinstance(node, Tree):          # Tree aithe adi entity
            name = " ".join(word for word, _ in node.leaves())
            entities.append((name, node.label()))
    return entities

print("\nEntities:")
for name, label in get_entities(tree):
    print(f"  {name:25s} {label}")

print("\nbinary=True:")
print(ne_chunk(pos_tag(word_tokenize(text)), binary=True))
# anni entities "NE" ane label tho vastayi

# Disaster text for CrisisBridge project:

disaster = ("Heavy rains flooded Chennai and Kanchipuram. "
            "The NDRF and Indian Army rescued 200 people near Tambaram.")
print("\nDisaster text entities:")
for name, label in get_entities(ne_chunk(pos_tag(word_tokenize(disaster)))):
    print(f"  {name:25s} {label}")
# "200 people" lanti numbers ni NLTK NER pattukodu, CD tag tho vidiga teeyali

# LEO MEEEHHHHSIIIII example ----------
messi = ("Lionel Messi left Barcelona in 2021 and joined Paris Saint-Germain. "
         "In 2023 he moved to Inter Miami in the United States.")
print("\nMessi entities:")
for name, label in get_entities(ne_chunk(pos_tag(word_tokenize(messi)))):
    print(f"  {name:25s} {label}")


lower = text.lower()
print("\nLowercase text entities:", get_entities(ne_chunk(pos_tag(word_tokenize(lower)))))

