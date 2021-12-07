# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:20:38 2021

@author: Salwa
"""
import nltk 
import numpy as np
import matplotlib.pyplot as plt 
#we gonna use a lematizer 
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD

#initilize the lemmatizer 
wordnet_lemmatizer = WordNetLemmatizer()
#read data 
titles = [line.rstrip() for line in open('all_book_titles.txt')]
#stopwords  
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
#from nltk.corpus import stopwords
#stopwords.words('english')
#tokenizer 
def my_tokenizer(s):
    s = s.lower() # downcase
    tokens = nltk.tokenize.word_tokenize(s) # split string into words (tokens)
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # put words into base form
    tokens = [t for t in tokens if t not in stopwords] # remove stopwords
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)] # remove any digits, i.e. "3rd edition"
    return tokens
#find the index for each word by going trough the antir vocabulary 
word_index_map = {}
current_index = 0
all_tokens = []
all_titles = []
index_word_map = []
for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except:
        pass

#there is no labels it's an unsupervised learning 
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        i = word_index_map[t]
        x[i] = 1
    return x
#SVD  input matrix 
N = len(all_tokens)
D = len(word_index_map)
X = np.zeros((D, N))
i = 0
for tokens in  all_tokens:
    X[:,i] = tokens_to_vector(tokens)
    i += 1
    
#initiate svd object 
svd = TruncatedSVD()
Z = svd.fit_transform(X)
plt.scatter(Z[:,0], Z[:,1])
for i in range(D):
    plt.annotate(s=index_word_map[i], xy=(Z[i,0], Z[i,1]))
plt.show()