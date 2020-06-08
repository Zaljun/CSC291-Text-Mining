from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 

import re
import numpy as np
import pandas as pd
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim import models,matutils
import spacy
from pprint import pprint

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics 
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter 

'''
def my_tokenizer(doc):
    
    all_tokens   = word_tokenize(doc)  
    all_pos_tags = pos_tag(all_tokens) 
 
    tokens_no_punct = [(t, pos) for (t,pos) in all_pos_tags if len(pos) > 1]  
    no_punkt_pos = []
    for (t,pos) in tokens_no_punct:
        valid = False
        for ch in t:
            if ch.isalnum():
                valid = True
        if valid:
            no_punkt_pos.append((t,pos))

    lower_tokens = [(t.lower(),pos) for (t,pos) in no_punkt_pos]
    
    porter = PorterStemmer()
    stemmed_tokens = [(porter.stem(t),pos) for (t, pos) in lower_tokens]
    
    stoplist = stopwords.words('english')
    stoplist.extend(["wine"])
    no_stopwords = [(t, pos) for (t, pos) in stemmed_tokens if t not in stoplist]
    
    good_tokens = [ t for (t,pos) in no_stopwords]
 
    return good_tokens
'''

dir_file = os.getcwd() # returns path to current directory
files_dir = os.listdir(dir_file)  # list of files in current directory

csv_files = [f for f in files_dir if f.endswith('csv')]
wine_file = csv_files[0]

fid = open(wine_file)
wine_df = pd.read_csv(wine_file)
wine_df.info  # the columns

# find unique variety
variety_dict = Counter(wine_df['variety'])     
most_common = [t[0] for t in variety_dict.most_common(20)]
print(variety_dict.most_common(20))
# main corpus = top 20 variety wine description
variety_top_20 = wine_df['variety'].isin(most_common)   # returns a bool index
print(variety_top_20.shape)
selected_wine = wine_df[variety_top_20]
print(selected_wine.shape)

stop_words = stopwords.words('english')
#stop_words.extend(['from','subject','re','edu','use'])

df = selected_wine
#print(df.target_names.unique())
print(df.head())

data = df.description.values.tolist()
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub('\s+', ' ' , sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]
print(data[:1])

def sent_to_words(sent):
    for sentence in sent:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc = True))

data_words = list(sent_to_words(data))
print(data_words[:1])

# build bigram and trigram model
bigram  = gensim.models.Phrases(data_words, min_count = 5, threshold = 100)
trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)
bigram_mod  = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram) 
print(trigram_mod[bigram_mod[data_words[0]]])

# funtion for stopwords, bigram, trigram and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigram(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags = ['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


data_words_nostop  = remove_stopwords(data_words)
data_words_bigrams = make_bigrams(data_words_nostop)

nlp = spacy.load('en', disable = ['parser', 'ner'])

data_lemmatized = lemmatization(data_words_bigrams, allowed_postags = ['NOUN', 'ADJ', 'ADV', 'VERB'])
print(data_lemmatized[:1])

def preprocessor(doc):
    data_words_nostop  = remove_stopwords(doc)
    data_words_bigrams = make_bigrams(data_words_nostop)
    data_lemmatized = lemmatization(data_words_bigrams, allowed_postags = ['NOUN', 'ADJ', 'ADV', 'VERB'])
    return data_lemmatized
# do clustering using minibatch K-means
#vect = CountVectorizer(stop_words = 'english',lowercase = True, min_df = 10)
vect = CountVectorizer(tokenizer = preprocessor, min_df = 10)
counter= vect.fit_transform(selected_wine['description'])

transf = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = False) 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transf.fit_transform(counter)

k_clusters = 20
model = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=200, batch_size=5000, n_init = 10)
model.fit(tf_idf)

# clustering results
print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]  # sort and reverse
terms = vect.get_feature_names()

for i in range(k_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:  # print first 10 terms from the cluster
        print(' %s' % terms[ind])
    print()

# clustering score
variety = selected_wine.variety.copy()
variety = pd.Categorical(variety)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(variety, model.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(variety, model.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(variety, model.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(variety, model.labels_))
#print("Silhouette Coefficient: %0.3f"
 #     % metrics.silhouette_score(X, km.labels_, sample_size=1000))
 
# check
index_Chardonnay = selected_wine['variety'].isin(['Chardonnay'])

print(sorted(Counter(model.labels_[index_Chardonnay]).items(),key = 
             lambda x:(x[1], x[0]), reverse =True)) 
m = model.labels_[index_Chardonnay]
