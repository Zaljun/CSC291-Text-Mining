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

# find unique variety
variety_dict = Counter(wine_df['variety'])     
most_common = [t[0] for t in variety_dict.most_common(20)]

# vectrize
vect = CountVectorizer(stop_words = 'english',lowercase = True, min_df = 10)
#vect = CountVectorizer(tokenizer = my_tokenizer)
counter= vect.fit_transform(wine_df['description'])

transf = TfidfTransformer(norm = 'l2', use_idf = True, smooth_idf = True, sublinear_tf = False) 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transf.fit_transform(counter)

lda = LatentDirichletAllocation(n_components=20,random_state=0)
lda.fit(counter)
lda.transform(counter)
tf_feature_name = vect.get_feature_names()

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
print_top_words(lda,tf_feature_name,10)