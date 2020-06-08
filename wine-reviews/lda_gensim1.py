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
    
    #porter = PorterStemmer()
    #stemmed_tokens = [(porter.stem(t),pos) for (t, pos) in lower_tokens]
    lemmatizer = spacy.lang.en.English()  
    lemmatized_tokens = [(lemmatizer(t),pos) for (t,pos) in lower_tokens]
    
    stoplist = stopwords.words('english')
    stoplist.extend(["wine"])
    no_stopwords = [(t, pos) for (t, pos) in lemmatized_tokens if t not in stoplist]
    
    good_tokens = [ t for (t,pos) in no_stopwords]
 
    return good_tokens


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
stop_words.extend(['also','that',"'s"])

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
#data_lemmatized = lemmatization(data_words_nostop, allowed_postags = ['NOUN', 'ADJ', 'ADV', 'VERB'])
pprint(data_lemmatized[:1])
type(data_lemmatized)

# create dictionary
id2word = corpora.Dictionary(data_lemmatized)
# create corpus
texts   = data_lemmatized
# term document frequency
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
                                            passes = 5,
                                            alpha = 'auto',
                                            per_word_topics = True)
# lda_model.print_topics()
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
#pprint(doc_lda[0])

# tf-idf matrix
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lda = models.LdaModel(corpus_tfidf, id2word = id2word, num_topics = 20) 
corpus_lda = lda[corpus_tfidf]
lda_csc_matrix = matutils.corpus2csc(corpus_lda).transpose()  #gensim sparse matrix to scipy sparse matrix
lda_csc_matrix.shape


# top terms in docs
# use top terms in topics of each docs to form new vocabulary
# do tf-idf using new vocabulary
topics_terms = lda_model.state.get_lambda() 
#convert estimates to probability (sum equals to 1 per topic)
topics_terms_proba = np.apply_along_axis(lambda x: x/x.sum(),1,topics_terms)
# find the right word based on column index
words = [lda_model.id2word[i] for i in range(topics_terms_proba.shape[1])]
#put everything together
term = pd.DataFrame(topics_terms_proba,columns=words)

matrx = lda_csc_matrix * term

# clustering
k_clusters = 20
model = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=200, batch_size=5000, n_init = 10)

########### for topic
#model.fit(lda_csc_matrix)

########## for terms
model.fit(matrx)

# clustering results
print("Top topics cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]  # sort and reverse

###########  for topic
for i in range(k_clusters):
    top = []
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :20]:  # print 20 topics from the cluster
        top.append(ind)
    print(top)

########## for terms
for i in range(k_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:  # print first 10 terms from the cluster
        print(' %s' % words[ind])
    print()

# clustering score
variety = selected_wine.variety.copy()
variety = pd.Categorical(variety)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(variety, model.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(variety, model.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(variety, model.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(variety, model.labels_))
#Sum of squared distances of samples to their closest cluster center.
print(model.inertia_/20) #266.3540622531673  k = 20 
                         #359.53637930976436 k = 10 
                         
    
'''                     
#centroids = model.cluster_centers_
#vect = CountVectorizer(stop_words = 'english', min_df = 10)
vect = CountVectorizer(tokenizer = my_tokenizer, min_df = 10)
counter= vect.fit_transform(selected_wine['description'])
transf = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = False) 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transf.fit_transform(counter)
model2 = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=200, batch_size=5000, n_init = 10)
model2.fit(tf_idf)
'''

'''
Sample:
    

Top 20 wines:    
[('Pinot Noir', 13272), ('Chardonnay', 11753), ('Cabernet Sauvignon', 9472), 
('Red Blend', 8946), ('Bordeaux-style Red Blend', 6915), ('Riesling', 5189), 
('Sauvignon Blanc', 4967), ('Syrah', 4142), ('Rosé', 3564), ('Merlot', 3102), 
('Nebbiolo', 2804), ('Zinfandel', 2714), ('Sangiovese', 2707), ('Malbec', 2652), 
('Portuguese Red', 2466), ('White Blend', 2360), ('Sparkling Blend', 2153), 
('Tempranillo', 1810), ('Rhône-style Red Blend', 1471), ('Pinot Gris', 1455)]

Shape of selected documents:
(93914, 14)

LDA topic extraction
topics:
############################# with no extension of stoplist:
[(0,
  '0.139*"cola" + 0.102*"alcohol" + 0.091*"high" + 0.066*"pair" + '
  '0.064*"grill" + 0.059*"streak" + 0.057*"peppery" + 0.052*"meat" + '
  '0.051*"olive" + 0.038*"intensity"'),
 (1,
  '0.136*"also" + 0.059*"whiff" + 0.058*"candy" + 0.050*"tone" + '
  '0.046*"element" + 0.045*"lean" + 0.040*"thick" + 0.040*"tangerine" + '
  '0.038*"carry" + 0.034*"lend"'),
 (2,
  '0.209*"blackberry" + 0.085*"spicy" + 0.085*"tannic" + 0.071*"cassis" + '
  '0.062*"smoky" + 0.048*"concentrate" + 0.047*"that" + 0.044*"blueberry" + '
  '0.040*"deep" + 0.040*"hard"'),
 (3,
  '0.122*"pepper" + 0.108*"white" + 0.085*"syrah" + 0.062*"time" + '
  '0.057*"cinnamon" + 0.052*"easy" + 0.049*"young" + 0.038*"flower" + '
  '0.034*"mint" + 0.033*"drinking"'),
 (4,
  '0.139*"fine" + 0.124*"style" + 0.122*"wood" + 0.091*"new" + 0.062*"lightly" '
  '+ 0.061*"toasty" + 0.058*"lot" + 0.057*"dominate" + 0.030*"source" + '
  '0.028*"pie"'),
 (5,
  '0.183*"finish" + 0.111*"note" + 0.075*"plum" + 0.066*"nose" + '
  '0.041*"chocolate" + 0.035*"medium" + 0.029*"long" + 0.025*"herbal" + '
  '0.023*"tobacco" + 0.023*"savory"'),
 (6,
  '0.077*"need" + 0.070*"strong" + 0.070*"develop" + 0.065*"quite" + '
  '0.064*"attractive" + 0.062*"pretty" + 0.060*"end" + 0.056*"oaky" + '
  '0.047*"cut" + 0.045*"winery"'),
 (7,
  '0.254*"currant" + 0.146*"leather" + 0.130*"dense" + 0.098*"bake" + '
  '0.093*"cedar" + 0.067*"jammy" + 0.061*"sangiovese" + 0.056*"soften" + '
  '0.043*"grip" + 0.014*"syrupy"'),
 (8,
  '0.225*"blend" + 0.205*"cabernet" + 0.138*"sauvignon" + 0.115*"merlot" + '
  '0.054*"complexity" + 0.051*"little" + 0.050*"franc" + 0.030*"caramel" + '
  '0.030*"petit_verdot" + 0.028*"lovely"'),
 (9,
  '0.207*"tart" + 0.161*"bit" + 0.137*"cranberry" + 0.129*"barrel" + '
  '0.117*"seem" + 0.098*"follow" + 0.060*"variety" + 0.022*"flat" + '
  '0.013*"fall" + 0.013*"grainy"'),
 (10,
  '0.275*"bottle" + 0.197*"add" + 0.183*"grapefruit" + 0.104*"blanc" + '
  '0.035*"grassy" + 0.031*"gooseberry" + 0.030*"interest" + 0.026*"passion" + '
  '0.020*"kiwi" + 0.014*"graham_cracker"'),
 (11,
  '0.121*"apple" + 0.082*"citrus" + 0.075*"green" + 0.073*"pear" + '
  '0.067*"peach" + 0.058*"chardonnay" + 0.050*"orange" + 0.040*"clean" + '
  '0.037*"pineapple" + 0.037*"honey"'),
 (12,
  '0.204*"lime" + 0.081*"much" + 0.066*"riesl" + 0.063*"rather" + '
  '0.049*"blossom" + 0.041*"intensely" + 0.041*"ample" + 0.037*"astringent" + '
  '0.034*"roasted" + 0.032*"rustic"'),
 (13,
  '0.239*"feel" + 0.175*"great" + 0.129*"big" + 0.106*"balanced" + '
  '0.066*"peel" + 0.062*"drive" + 0.048*"refresh" + 0.043*"extract" + '
  '0.021*"become" + 0.015*"direct"'),
 (14,
  '0.096*"wine" + 0.086*"flavor" + 0.076*"fruit" + 0.041*"acidity" + '
  '0.039*"drink" + 0.033*"ripe" + 0.032*"dry" + 0.024*"oak" + 0.023*"show" + '
  '0.022*"rich"'),
 (15,
  '0.154*"come" + 0.143*"lead" + 0.112*"mouth" + 0.080*"wild" + 0.063*"tangy" '
  '+ 0.054*"back" + 0.042*"together" + 0.040*"right" + 0.036*"next" + '
  '0.032*"cab"'),
 (16,
  '0.189*"s" + 0.112*"there" + 0.100*"elegant" + 0.079*"core" + '
  '0.066*"powerful" + 0.039*"sense" + 0.038*"skin" + 0.033*"bordeaux" + '
  '0.024*"wrap" + 0.023*"reserve"'),
 (17,
  '0.082*"aroma" + 0.080*"tannin" + 0.080*"cherry" + 0.076*"palate" + '
  '0.071*"black" + 0.050*"red" + 0.049*"spice" + 0.042*"berry" + 0.032*"offer" '
  '+ 0.031*"dark"'),
 (18,
  '0.192*"clove" + 0.145*"simple" + 0.137*"crush" + 0.102*"anise" + '
  '0.097*"cocoa" + 0.088*"enjoy" + 0.056*"graphite" + 0.046*"rose_petal" + '
  '0.037*"fairly" + 0.024*"iron"'),
 (19,
  '0.307*"lemon" + 0.138*"creamy" + 0.133*"malbec" + 0.079*"appeal" + '
  '0.066*"quality" + 0.044*"reveal" + 0.043*"gris" + 0.030*"nut" + '
  '0.027*"small" + 0.019*"satisfy"')]
 
############################# after modifying the stoplist:
[(0,
  '0.118*"layer" + 0.096*"time" + 0.083*"clove" + 0.079*"almost" + '
  '0.071*"strong" + 0.064*"glass" + 0.062*"concentrated" + 0.044*"anise" + '
  '0.041*"chewy" + 0.040*"vibrant"'),
 (1,
  '0.164*"black" + 0.088*"blend" + 0.080*"cabernet" + 0.070*"dark" + '
  '0.054*"sauvignon" + 0.046*"chocolate" + 0.045*"merlot" + 0.038*"currant" + '
  '0.032*"licorice" + 0.028*"cassis"'),
 (2,
  '0.082*"apple" + 0.076*"crisp" + 0.058*"lemon" + 0.055*"citrus" + '
  '0.050*"green" + 0.050*"pear" + 0.047*"tart" + 0.044*"lime" + '
  '0.039*"chardonnay" + 0.035*"mineral"'),
 (3,
  '0.191*"syrah" + 0.120*"clean" + 0.111*"grape" + 0.075*"mint" + '
  '0.064*"blanc" + 0.059*"take" + 0.056*"streak" + 0.055*"brisk" + '
  '0.050*"almond" + 0.045*"nuance"'),
 (4,
  '0.154*"lead" + 0.106*"cranberry" + 0.094*"coffee" + 0.076*"follow" + '
  '0.071*"espresso" + 0.069*"violet" + 0.052*"sour" + 0.037*"grain" + '
  '0.033*"bean" + 0.022*"forest_floor"'),
 (5,
  '0.195*"s" + 0.174*"toast" + 0.143*"style" + 0.116*"there" + 0.080*"bake" + '
  '0.051*"cut" + 0.047*"caramel" + 0.046*"soften" + 0.018*"california" + '
  '0.016*"buttered"'),
 (6,
  '0.075*"oak" + 0.069*"rich" + 0.061*"blackberry" + 0.053*"good" + '
  '0.047*"well" + 0.044*"balance" + 0.044*"give" + 0.041*"year" + '
  '0.038*"vineyard" + 0.030*"fine"'),
 (7,
  '0.123*"white" + 0.101*"peach" + 0.059*"easy" + 0.057*"grapefruit" + '
  '0.053*"round" + 0.051*"stone" + 0.047*"attractive" + 0.043*"melon" + '
  '0.043*"flower" + 0.038*"drinking"'),
 (8,
  '0.343*"come" + 0.151*"not" + 0.128*"do" + 0.090*"nice" + 0.082*"impressive" '
  '+ 0.035*"try" + 0.028*"barolo" + 0.026*"smoked_meat" + 0.022*"stay" + '
  '0.017*"early"'),
 (9,
  '0.251*"open" + 0.137*"deliver" + 0.110*"suggest" + 0.086*"delicate" + '
  '0.074*"enjoy" + 0.064*"baking" + 0.050*"reveal" + 0.049*"soon" + '
  '0.048*"straightforward" + 0.031*"small"'),
 (10,
  '0.333*"pepper" + 0.204*"bottle" + 0.115*"quite" + 0.068*"sip" + 0.064*"wet" '
  '+ 0.048*"ample" + 0.025*"golden" + 0.024*"intriguing" + 0.022*"mingle" + '
  '0.011*"jasmine"'),
 (11,
  '0.223*"wine" + 0.176*"fruit" + 0.100*"tannin" + 0.094*"acidity" + '
  '0.090*"drink" + 0.076*"ripe" + 0.033*"structure" + 0.030*"firm" + '
  '0.028*"juicy" + 0.024*"fruity"'),
 (12,
  '0.052*"young" + 0.050*"franc" + 0.044*"acid" + 0.044*"mix" + 0.043*"pretty" '
  '+ 0.041*"tangy" + 0.041*"whiff" + 0.040*"yet" + 0.031*"winery" + '
  '0.030*"mature"'),
 (13,
  '0.240*"still" + 0.178*"need" + 0.163*"develop" + 0.100*"first" + '
  '0.054*"interesting" + 0.051*"mushroom" + 0.047*"begin" + 0.024*"pleasure" + '
  '0.023*"dominant" + 0.022*"otherwise"'),
 (14,
  '0.140*"taste" + 0.116*"color" + 0.102*"edge" + 0.099*"body" + 0.078*"end" + '
  '0.075*"concentration" + 0.052*"midpalate" + 0.045*"sweetness" + '
  '0.042*"grow" + 0.035*"bite"'),
 (15,
  '0.117*"great" + 0.097*"linger" + 0.071*"balanced" + 0.064*"crush" + '
  '0.060*"textur" + 0.052*"enough" + 0.048*"pair" + 0.047*"elegance" + '
  '0.042*"length" + 0.042*"quality"'),
 (16,
  '0.105*"flavor" + 0.059*"finish" + 0.055*"aroma" + 0.053*"cherry" + '
  '0.051*"palate" + 0.039*"dry" + 0.036*"note" + 0.032*"spice" + 0.029*"show" '
  '+ 0.028*"berry"'),
 (17,
  '0.185*"full" + 0.147*"soft" + 0.146*"texture" + 0.138*"bodied" + '
  '0.087*"character" + 0.071*"smooth" + 0.046*"ready" + 0.045*"generous" + '
  '0.037*"intense" + 0.018*"rounded"'),
 (18,
  '0.115*"spicy" + 0.076*"smoke" + 0.075*"seem" + 0.072*"wild" + 0.071*"core" '
  '+ 0.063*"that" + 0.054*"hard" + 0.048*"back" + 0.043*"appeal" + '
  '0.041*"pack"'),
 (19,
  '0.595*"fresh" + 0.192*"earth" + 0.057*"turn" + 0.054*"citrusy" + '
  '0.037*"dust" + 0.000*"age" + 0.000*"lively" + 0.000*"rise" + '
  '0.000*"delicious" + 0.000*"freshness"')]

#################################################################
 
=================== Top topics cluster: (k = 20)
Cluster 0:
[1, 4, 6, 14, 3, 13, 15, 0, 12, 16]
Cluster 1:
[17, 15, 0, 10, 13, 7, 19, 6, 3, 2]
Cluster 2:
[16, 6, 8, 19, 15, 12, 9, 17, 14, 13]
Cluster 3:
[10, 0, 15, 5, 8, 9, 3, 13, 19, 7]
Cluster 4:
[14, 4, 1, 16, 13, 15, 3, 18, 9, 11]
Cluster 5:
[2, 5, 8, 0, 14, 18, 3, 15, 6, 17]
Cluster 6:
[8, 9, 5, 4, 10, 2, 0, 16, 18, 15]
Cluster 7:
[16, 19, 17, 12, 14, 15, 9, 7, 3, 5]
Cluster 8:
[6, 17, 15, 13, 14, 4, 8, 0, 1, 10]
Cluster 9:
[13, 8, 14, 17, 16, 0, 6, 15, 1, 3]
Cluster 10:
[16, 14, 17, 19, 15, 10, 12, 5, 9, 13]
Cluster 11:
[4, 3, 8, 14, 15, 1, 16, 0, 9, 2]
Cluster 12:
[6, 13, 4, 14, 1, 15, 0, 3, 18, 17]
Cluster 13:
[10, 2, 0, 5, 6, 8, 3, 7, 19, 15]
Cluster 14:
[0, 10, 3, 2, 6, 13, 8, 17, 14, 15]
Cluster 15:
[5, 2, 15, 8, 0, 14, 17, 16, 3, 13]
Cluster 16:
[10, 0, 15, 8, 17, 6, 13, 5, 16, 3]
Cluster 17:
[15, 8, 17, 10, 13, 9, 14, 4, 6, 16]
Cluster 18:
[17, 15, 10, 0, 6, 8, 13, 16, 14, 2]
Cluster 19:
[14, 13, 6, 4, 1, 17, 3, 16, 15, 5]

Homogeneity: 0.214
Completeness: 0.212
V-measure: 0.213
Adjusted Rand-Index: 0.092

===================Top topics cluster:  (k = 10)
Cluster 0:
[6, 14, 4, 13, 1, 17, 15, 16, 8, 0]
Cluster 1:
[16, 19, 17, 14, 15, 6, 12, 9, 5, 10]
Cluster 2:
[17, 15, 6, 10, 0, 16, 13, 8, 2, 14]
Cluster 3:
[10, 15, 0, 8, 2, 17, 5, 6, 13, 3]
Cluster 4:
[8, 9, 5, 4, 10, 0, 2, 16, 17, 18]
Cluster 5:
[14, 4, 13, 1, 16, 6, 3, 5, 17, 15]
Cluster 6:
[15, 5, 1, 4, 16, 3, 8, 9, 17, 6]
Cluster 7:
[13, 1, 14, 8, 16, 17, 6, 4, 5, 0]
Cluster 8:
[2, 5, 8, 10, 14, 0, 16, 17, 18, 6]
Cluster 9:
[0, 10, 3, 6, 2, 13, 17, 8, 1, 5]

Homogeneity: 0.196
Completeness: 0.245
V-measure: 0.218
Adjusted Rand-Index: 0.116
'''
'''
##############################################################
Use matrix = Document-topic * topic-term for clustering

Top 10 terms per cluster:
    
 Cluster 0:
 young
 franc
 acid
 mix
 pretty
 tangy
 whiff
 yet
 winery
 mature

Cluster 1:
 come
 fresh
 not
 do
 nice
 impressive
 earth
 try
 barolo
 smoked_meat

Cluster 2:
 great
 linger
 apple
 crisp
 balanced
 crush
 textur
 lemon
 citrus
 enough

Cluster 3:
 pepper
 bottle
 taste
 color
 edge
 body
 quite
 end
 concentration
 midpalate

Cluster 4:
 full
 soft
 texture
 bodied
 character
 smooth
 ready
 generous
 intense
 fresh

Cluster 5:
 come
 white
 peach
 not
 do
 easy
 grapefruit
 round
 nice
 stone

Cluster 6:
 pepper
 bottle
 quite
 sip
 wet
 ample
 golden
 intriguing
 mingle
 fresh

Cluster 7:
 taste
 color
 edge
 body
 end
 concentration
 midpalate
 sweetness
 grow
 bite

Cluster 8:
 come
 not
 do
 nice
 impressive
 try
 barolo
 smoked_meat
 stay
 still

Cluster 9:
 apple
 crisp
 lemon
 citrus
 green
 pear
 tart
 lime
 chardonnay
 mineral

Cluster 10:
 pepper
 bottle
 quite
 sip
 wet
 ample
 fresh
 open
 golden
 intriguing

Cluster 11:
 great
 linger
 balanced
 crush
 textur
 enough
 pair
 elegance
 length
 quality

Cluster 12:
 layer
 time
 clove
 almost
 strong
 glass
 concentrated
 anise
 chewy
 vibrant

Cluster 13:
 come
 not
 do
 nice
 impressive
 try
 barolo
 smoked_meat
 stay
 early

Cluster 14:
 come
 not
 do
 nice
 impressive
 full
 soft
 texture
 bodied
 try

Cluster 15:
 great
 linger
 balanced
 crush
 textur
 enough
 pair
 elegance
 length
 quality

Cluster 16:
 white
 peach
 easy
 grapefruit
 round
 stone
 attractive
 melon
 flower
 drinking

Cluster 17:
 come
 not
 do
 apple
 crisp
 nice
 lemon
 citrus
 impressive
 green

Cluster 18:
 fresh
 earth
 come
 turn 
 citrusy
 not
 do
 pepper
 dust
 taste

Cluster 19:
 pepper
 bottle
 quite
 sip
 wet
 ample
 great
 linger
 white
 fresh

Homogeneity: 0.201
Completeness: 0.200
V-measure: 0.200
Adjusted Rand-Index: 0.089
model.inertia_ = 414.98636713365056 (unnormalized)
(#words = 23413)(include bigrams)
'''