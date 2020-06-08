
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 

import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics 


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


dir_file = os.getcwd() # returns path to current directory
files_dir = os.listdir(dir_file)  # list of files in current directory

csv_files = [f for f in files_dir if f.endswith('csv')]
wine_file = csv_files[0]

fid = open(wine_file)
wine_df = pd.read_csv(wine_file)
wine_df.info  # the columns

vect = CountVectorizer(stop_words = 'english',lowercase = True, min_df = 10)
#vect = CountVectorizer(tokenizer = my_tokenizer)
counter= vect.fit_transform(wine_df['description'])

transf = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = False) 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transf.fit_transform(counter)

# cluster the wines using KMeans - way too slow
# cluster wines using MiniBatchKmeans
k_clusters = 20
model = MiniBatchKMeans(n_clusters=k_clusters, init='k-means++', max_iter=200, batch_size=5000, n_init = 10)
model.fit(tf_idf)
#print("\nSilhouette Coefficient: %0.3f" %metrics.silhouette_score(tf_idf, model.labels_, metric = "cosine"))

# use the variety column to check clustering
# find unique variety
from collections import Counter 
variety_dict = Counter(wine_df['variety'])     
most_common = [t[0] for t in variety_dict.most_common(20)]
# filter wines based on Variety
'''
variety_top_20 = wine_df['variety'].isin(most_common)   # returns a bool index
print(variety_top_20.shape)
selected_wine = wine_df[variety_top_20]
print(selected_wine.shape)
'''
#set_variety = set(wine_df['variety'])
#print(len(set_variety))

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]  # sort and reverse
terms = vect.get_feature_names()

for i in range(k_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:  # print first k terms from the cluster
        print(' %s' % terms[ind])
    print()

'''
variety = selected_wine.variety.copy()
variety = pd.Categorical(variety)'''
#print("\n")
#print("Prediction")
 
# clusterering evaluation
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, model.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(labels, model.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(labels, model.labels_))
print("Adjusted Rand-Index: %.3f"
# metrics.adjusted_rand_score(labels, km.labels_))
'''
Top terms per cluster:
Cluster 0:
 oak
 new
 french
 thi
 age
Cluster 1:
 pinot
 noir
 cherri
 thi
 flavor
Cluster 2:
 like
 tast
 smell
 thi
 flavor
Cluster 3:
 crisp
 fruiti
 acid
 bright
 fresh
Cluster 4:
 fruiti
 soft
 attract
 drink
 readi
Cluster 5:
 wood
 age
 fruit
 ha
 thi
Cluster 6:
 alongsid
 cherri
 red
 palat
 tannin
Cluster 7:
 feel
 finish
 aroma
 flavor
 blackberri
Cluster 8:
 textur
 flavor
 bodi
 thi
 cherri
Cluster 9:
 orang
 thi
 acid
 flavor
 ha
Cluster 10:
 herbal
 berri
 finish
 flavor
 aroma
Cluster 11:
 ripe
 rich
 full
 fruit
 drink
Cluster 12:
 blanc
 sauvignon
 thi
 flavor
 blend
Cluster 13:
 espresso
 palat
 toast
 oak
 tannin
Cluster 14:
 structur
 fruit
 tannin
 firm
 dens
Cluster 15:
 black
 tannin
 cherri
 palat
 licoric
Cluster 16:
 nose
 black
 palat
 bottl
 show
Cluster 17:
 pair
 meat
 thi
 spice
 bright
Cluster 18:
 berri
 plum
 aroma
 finish
 flavor
Cluster 19:
 white
 flower
 appl
 palat
 yellow
Cluster 20:
 red
 strawberri
 fruit
 thi
 tannin
Cluster 21:
 herb
 aroma
 barrel
 flavor
 fruit
Cluster 22:
 cabernet
 sauvignon
 merlot
 blend
 franc
Cluster 23:
 dri
 cherri
 cola
 flavor
 raspberri
Cluster 24:
 alcohol
 high
 flavor
 hot
 thi
Cluster 25:
 blackberri
 currant
 year
 cabernet
 tannin
Cluster 26:
 appl
 palat
 fresh
 citru
 finish
Cluster 27:
 thi
 fruit
 finish
 dark
 black
Cluster 28:
 pineappl
 chardonnay
 butter
 sweet
 flavor
Cluster 29:
 peach
 thi
 finish
 flavor
 honey

 Top terms per cluster:
Cluster 0:
 citrus
 melon
 finish
 aromas
 flavors
Cluster 1:
 herbal
 berry
 finish
 flavors
 aromas
Cluster 2:
 wood
 aging
 wine
 fruit
 drink
Cluster 3:
 notes
 black
 finish
 fruit
 chocolate
Cluster 4:
 honey
 wine
 sweet
 apricot
 flavors
Cluster 5:
 cherry
 wine
 raspberry
 bright
 fruit
Cluster 6:
 apricot
 peach
 flavors
 wine
 orange
Cluster 7:
 cabernet
 sauvignon
 merlot
 blend
 franc
Cluster 8:
 plum
 berry
 aromas
 finish
 flavors
Cluster 9:
 apple
 pear
 flavors
 palate
 green
Cluster 10:
 red
 cherry
 berry
 palate
 tannins
Cluster 11:
 fruit
 oak
 flavors
 barrel
 aromas
Cluster 12:
 chardonnay
 buttered
 toast
 pineapple
 vanilla
Cluster 13:
 fermented
 stainless
 steel
 aged
 barrel
Cluster 14:
 pinot
 noir
 cherry
 flavors
 silky
Cluster 15:
 wine
 tannins
 fruits
 structure
 firm
Cluster 16:
 wine
 soft
 ready
 drink
 fruity
Cluster 17:
 flavors
 dry
 blackberries
 blackberry
 cherries
Cluster 18:
 blanc
 sauvignon
 flavors
 wine
 crisp
Cluster 19:
 white
 peach
 wine
 aromas
 citrus
Cluster 20:
 lemon
 lime
 riesling
 dry
 palate
Cluster 21:
 black
 tannins
 alongside
 palate
 cherry
Cluster 22:
 wine
 fruit
 tannins
 acidity
 character
Cluster 23:
 wine
 ripe
 fruits
 rich
 drink
Cluster 24:
 nose
 dried
 palate
 black
 bottling
Cluster 25:
 white
 palate
 yellow
 apple
 aromas
Cluster 26:
 crisp
 wine
 acidity
 fruity
 bright
Cluster 27:
 black
 cherry
 dark
 tobacco
 leather
Cluster 28:
 wine
 fruit
 flavors
 years
 vineyard
Cluster 29:
 bodied
 medium
 wine
 flavors
 cherry
 '''