from sklearn.feature_extraction.text import CountVectorizer


# corpus = [
#     "zebra apple ball cat cat",
#     "ball cat dog elephant",
#     "very very unique"
# ]

corpus = [
    "apple ball cat",
    "ball cat dog elephant",
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
# print(X.toarray())
# print(vectorizer.get_feature_names_out())


max_features = 4
ngrams = 2 # tri gram

vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, ngrams))
X = vectorizer.fit_transform(corpus)
print(X.toarray())
print(vectorizer.get_feature_names_out())

# corpus = [
#     "apple ball cat",
#     "ball cat dog",
# ]

# ['apple' 
# 'apple ball' 
# 'apple ball cat' 
# 'ball' 
# 'ball cat' 
# 'ball cat dog'
# 'cat' 
# 'cat dog' 
# 'dog']
