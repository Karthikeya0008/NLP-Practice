# Date: 04-08-2026
# Topic: Tokenization (sentence level and word level)
# Class lo cheppinattu, text ni chinna units (tokens) ga break cheyyadame tokenization

from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
from nltk.tokenize import RegexpTokenizer, TweetTokenizer

text = ("Dr. Sharma reached Vellore at 9 a.m. today. "
        "He said, \"NLP isn't easy!\" Then he left for Chennai. "
        "The ticket cost Rs. 450.50 only.")

#1.Sentence tokenization
# "Dr." and "a.m." daggara sentence split avvakudadu. 
sentences = sent_tokenize(text)
print("Sentences:")
for i, s in enumerate(sentences, 1):
    print(f"  {i}. {s}")


#2. Word tokenization
words = word_tokenize(text)
print("\nword_tokenize:", words)
# "isn't" ni "is" + "n't" ga split chesindi

# wordpunct_tokenize tho compare cheddam
print("\nwordpunct_tokenize:", wordpunct_tokenize(text))
# idi "isn" , "'", "t" ga vidagottindi -> punctuation meeda strict ga split chestundi

# 3. Python split() tho compare
print("\nsimple split():", text.split())
# split() lo "today." ani full stop word tho pate untundi, anduke NLTK tokenizer better

#. 4. Regex tokenizer.
# kevalam words matrame kavali, punctuation vaddu ante
regex_tok = RegexpTokenizer(r"\w+")
print("\nRegexpTokenizer(\\w+):", regex_tok.tokenize(text))

#. 5. Tweet tokenizer.
# social media text ki special tokenizer undi, hashtags and emojis ni vidagottadu
tweet = "Floods in #Chennai again!!! @NDRF please help :( stuck near Velachery"
print("\nword_tokenize on tweet :", word_tokenize(tweet))
print("TweetTokenizer on tweet:", TweetTokenizer().tokenize(tweet))
# word_tokenize lo "#" and "Chennai" separate ayyayi, TweetTokenizer lo "#Chennai" okate token

tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
print("\nTotal words (alphabets only):", len(tokens))
print("Unique words:", len(set(tokens)))

#7 Messi example
# naa favourite player!
messi = ("Lionel Messi won the 2022 FIFA World Cup with Argentina. "
         "He scored twice in the final vs. France! Messi's 8th Ballon d'Or came in 2023.")
print("\nMessi sentences:")
for s in sent_tokenize(messi):
    print("  ", s)
print("Messi words:", word_tokenize(messi))
# "vs." daggara sentence break avvakudadu, "Messi's" ni "Messi" + "'s" ga split chestundi
# "d'Or" ela tokenize avutundo chudali, French word kabatti konchem tricky

messi_tweet = "GOAT  @TeamMessi scores again!!! #Messi #InterMiami"
print("Messi tweet:", TweetTokenizer().tokenize(messi_tweet))
# emoji and hashtags anni separate tokens ga correct ga vachayi

