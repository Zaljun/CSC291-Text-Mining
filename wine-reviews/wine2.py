
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
import spacy

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from sklearn import metrics 
from collections import Counter 
from sklearn.decomposition import LatentDirichletAllocation

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
    stoplist.extend(["'s","wine"])
    #no_stopwords = [(t, pos) for (t, pos) in stemmed_tokens if t not in stoplist]
    no_stopwords = [(t, pos) for (t, pos) in lemmatized_tokens if t not in stoplist]
    
    good_tokens = [ t for (t,pos) in no_stopwords]
 
    return good_tokens


dir_file = os.getcwd() # returns path to current directory
files_dir = os.listdir(dir_file)  # list of files in current directory
csv_files = [f for f in files_dir if f.endswith('csv')]
wine_file = csv_files[0]

fid = open(wine_file)
wine_df = pd.read_csv(wine_file)

print(wine_df.columns)  # the columns
print(wine_df.shape)

# extract top 20 variety wine
# filter based on variety
variety_dict = Counter(wine_df['variety'])     
most_common = [t[0] for t in variety_dict.most_common(20)]
print(variety_dict.most_common(20))

variety_top_20 = wine_df['variety'].isin(most_common)   # returns a bool index
print(variety_top_20.shape)
selected_wine = wine_df[variety_top_20]
print(selected_wine.shape)

# do clustering using minibatch K-means
vect = CountVectorizer(tokenizer = my_tokenizer, min_df = 10)
counter= vect.fit_transform(selected_wine['description'])
transf = TfidfTransformer(norm = 'l2', use_idf = True, 
                          smooth_idf = True, sublinear_tf = False) 

# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transf.fit_transform(counter)

k_clusters = 20
model_b = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', 
                          max_iter=200, batch_size=5000, n_init = 10)
model_b.fit(tf_idf)


# clustering results
print("Top terms per cluster:")
order_centroids = model_b.cluster_centers_.argsort()[:, ::-1]  # sort and reverse
terms = vect.get_feature_names()


for i in range(k_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:  # print first 10 terms from the cluster
        print(' %s' % terms[ind])
    print()

# clustering score
variety = selected_wine.variety.copy()
variety = pd.Categorical(variety)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(variety, model_b.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(variety, model_b.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(variety, model_b.labels_))
print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(variety, model_b.labels_))
#Sum of squared distances of samples to their closest cluster center.
print(model_b.inertia_/6349) 
                           #13.740690835658736 k = 20
                           #13.896722311217765 k = 10 
                           #13.797526864305791 k = 15
                           #13.705306867637496 k = 25
                           #14.02656171320037 k = 5
                           #13.65176873289632  k = 30

# check
index_Chardonnay = selected_wine['variety'].isin(['Chardonnay'])

print(sorted(Counter(model_b.labels_[index_Chardonnay]).items(),key = 
             lambda x:(x[1], x[0]), reverse =True)) 
m = model_b.labels_[index_Chardonnay]


'''
Sample:
    
Columns of dataset:   
 Index(['Unnamed: 0', 'country', 'description', 'designation', 'points',
       'price', 'province', 'region_1', 'region_2', 'taster_name',
       'taster_twitter_handle', 'title', 'variety', 'winery'],
      dtype='object')
 
Shape of total documents:
(129971, 14)

Top 20 varieties:
[('Pinot Noir', 13272), ('Chardonnay', 11753), ('Cabernet Sauvignon', 9472), 
('Red Blend', 8946), ('Bordeaux-style Red Blend', 6915), ('Riesling', 5189), 
('Sauvignon Blanc', 4967), ('Syrah', 4142), ('Rosé', 3564), ('Merlot', 3102), 
('Nebbiolo', 2804), ('Zinfandel', 2714), ('Sangiovese', 2707), ('Malbec', 2652), 
('Portuguese Red', 2466), ('White Blend', 2360), ('Sparkling Blend', 2153), 
('Tempranillo', 1810), ('Rhône-style Red Blend', 1471), ('Pinot Gris', 1455)]

Shape of selected documents:
(93914, 14)

Top terms per cluster:  (k = 20)
Cluster 0:
 blackberri
 thi
 flavor
 tannin
 currant
 dri
 oak
 black
 tannic
 rich

Cluster 1:
 soft
 thi
 flavor
 cherri
 ripe
 tannin
 fruit
 sweet
 drink
 blackberri

Cluster 2:
 fruit
 ripe
 tannin
 rich
 acid
 drink
 thi
 black
 balanc
 well

Cluster 3:
 appl
 lemon
 pear
 palat
 thi
 fresh
 finish
 nose
 flavor
 citru

Cluster 4:
 black
 cherri
 thi
 nose
 palat
 red
 flavor
 plum
 fruit
 show

Cluster 5:
 pinot
 noir
 cherri
 raspberri
 flavor
 silki
 thi
 cola
 acid
 dri

Cluster 6:
 dens
 structur
 fruit
 tannin
 power
 age
 concentr
 firm
 thi
 dark

Cluster 7:
 palat
 espresso
 tannin
 aroma
 alongsid
 offer
 toast
 dri
 black
 cherri

Cluster 8:
 citru
 tropic
 blanc
 thi
 fruit
 flavor
 finish
 sauvignon
 melon
 chardonnay

Cluster 9:
 palat
 cherri
 tannin
 alongsid
 berri
 black
 aroma
 red
 offer
 pepper

Cluster 10:
 barrel
 aroma
 fruit
 flavor
 spice
 herb
 note
 coffe
 vanilla
 finish

Cluster 11:
 fruiti
 acid
 crisp
 drink
 fresh
 readi
 attract
 thi
 ha
 bright

Cluster 12:
 sweet
 flavor
 tast
 candi
 like
 thi
 cherri
 finish
 aroma
 ha

Cluster 13:
 age
 fruit
 ha
 drink
 thi
 wood
 ripe
 charact
 acid
 textur

Cluster 14:
 light
 crisp
 fresh
 acid
 thi
 ha
 fruit
 charact
 attract
 flavor

Cluster 15:
 cabernet
 sauvignon
 merlot
 blend
 franc
 petit
 verdot
 thi
 black
 cherri

Cluster 16:
 berri
 finish
 aroma
 plum
 flavor
 herbal
 feel
 palat
 thi
 note

Cluster 17:
 tart
 thi
 fruit
 flavor
 cranberri
 light
 strawberri
 finish
 cherri
 acid

Cluster 18:
 pineappl
 chardonnay
 butter
 flavor
 toast
 vanilla
 oak
 acid
 thi
 peach

Cluster 19:
 riesl
 peach
 finish
 miner
 lime
 thi
 acid
 dri
 lemon
 tangerin

Homogeneity: 0.317
Completeness: 0.301
V-measure: 0.309
Adjusted Rand-Index: 0.139

####################################################
Top terms per cluster:  (k = 10)
Cluster 0:
 flavor
 oak
 aroma
 thi
 citru
 fruit
 chardonnay
 finish
 toast
 vanilla

Cluster 1:
 cherri
 red
 palat
 thi
 nose
 black
 show
 aroma
 spice
 pepper

Cluster 2:
 blackberri
 tannin
 thi
 black
 flavor
 fruit
 currant
 rich
 year
 oak

Cluster 3:
 appl
 lemon
 thi
 flavor
 peach
 lime
 acid
 finish
 palat
 pear

Cluster 4:
 fruit
 drink
 ripe
 age
 rich
 thi
 ha
 structur
 acid
 tannin

Cluster 5:
 fruiti
 acid
 drink
 attract
 ha
 thi
 fresh
 light
 crisp
 soft

Cluster 6:
 cabernet
 plum
 aroma
 flavor
 finish
 berri
 blend
 thi
 sauvignon
 merlot

Cluster 7:
 pinot
 noir
 cherri
 raspberri
 flavor
 cola
 thi
 silki
 dri
 acid

Cluster 8:
 thi
 flavor
 fruit
 finish
 aroma
 textur
 bodi
 ha
 cherri
 sweet

Cluster 9:
 palat
 black
 tannin
 alongsid
 cherri
 aroma
 offer
 berri
 spice
 licoric
 
Homogeneity: 0.275
Completeness: 0.339
V-measure: 0.304
Adjusted Rand-Index: 0.148
'''