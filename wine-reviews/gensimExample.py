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
import spacy
from pprint import pprint

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics 
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter 

stop_words = stopwords.words('english')
#stop_words.extend(['from','subject','re','edu','use'])

df = pd.read_json('https://raw.githubusercontent.com/selva86/datasets/master/newsgroups.json')
print(df.target_names.unique())
print(df.head())

data = df.content.values.tolist()
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub('\s+', ' ' , sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]
print(data[:1])

def sent_to_words(sent):
    for sentence in sent:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc = True))

data_words = list(sent_to_words(data))
print(data_words[:1])

bigram  = gensim.models.Phrases(data_words, min_count = 5, threshold = 100)
trigram = gensim.models.Phrases(bigram[data_words], threshold = 100)
bigram_mod  = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram) 
print(trigram_mod[bigram_mod[data_words[0]]])

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

# create dictionary
id2word = corpora.Dictionary(data_lemmatized)
# create corpus
texts   = data_lemmatized
# termdocument frequency
corpus  = [id2word.doc2bow(text) for text in texts]
print(corpus[:1])
# readable corpus (term-frequency)
[[(id2word[id],freq) for id, freq in cp] for cp in corpus[:1]]

# build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus = corpus,
                                            id2word = id2word,
                                            num_topics = 20,
                                            random_state = 100,
                                            update_every = 1,
                                            chunksize = 100,
                                            passes = 10,
                                            alpha = 'auto',
                                            per_word_topics = True)
# lda_model.print_topics()
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]