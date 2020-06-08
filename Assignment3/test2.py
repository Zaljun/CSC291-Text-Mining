# IR
# Zhaojun Jia
# h702915924
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.corpus import reuters
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np

# nlp processing (unigram)
def process(text):
    all_tokens   = word_tokenize(text)  
    all_pos_tags = pos_tag(all_tokens) 
    
 # get tokens without punctuation
    tokens_no_punct = [(t, pos) for (t,pos) in all_pos_tags if len(pos) > 1]  
    no_punkt_pos    = []
    for (t,pos) in tokens_no_punct:
        valid = False
        for ch in t:
            if ch.isalnum():
                valid = True
        if valid:
            no_punkt_pos.append((t,pos))
 # lowering
    lower_tokens   = [(t.lower(),pos) for (t,pos) in no_punkt_pos]
    
 # stemming
    porter         = PorterStemmer()
    stemmed_tokens = [(porter.stem(t),pos) for (t, pos) in lower_tokens]
    
 # remove stop words
    stoplist       = stopwords.words('english')
    stoplist.extend(["n't", "'s"])
    no_stopwords   = [(t, pos) for (t, pos) in stemmed_tokens if t not in stoplist]
    
 # get tokens
    good_tokens    = [ t for (t,pos) in no_stopwords]
 
    return good_tokens

# get corpus
# some of the documents are the same
categories = reuters.categories()   # returns a list of all categories 
c_list     = reuters.fileids('trade')
doc_list   = [reuters.raw(t) for t in c_list]

"""
My dataset could be rueters news, of which category is "trade".

I choose trade news as dataset for several reasons:
    1. The scope of trade news could be vast. News corpus ofen cover multiple 
    topics even in one category. You can see different Countries, different 
    fields, different type of trade, etc. This makes it easier to choose a 
    query and see what happen when the query improves.
    
    2. The information of news is always simple and direct, which is easy for 
    me to read and justify the relevance.
    
    3. The number of trade news is big enough and the average length of it is 
    160, which won't be too short nor too long.
    
The very first query I plan to use is: Japan import transaction.

The number of files in this corpus is 485. The min length is 2 by tokens, while the max 
is 614. The average length is 160.
"""


# vectorizing
# generate tf-idf matrix of docs
vectorizer = CountVectorizer(tokenizer = process)
count_vect = vectorizer.fit_transform(doc_list)

# get vocabulary list
vocabulary = vectorizer.get_feature_names()
print('vocabulary size:', len(vocabulary), '\n')

# vocabulary counting list
count_v    = count_vect.toarray().sum(axis=0)

# list of term frequency
# by merging vocabulary and count_v
term_freq  = list(map(lambda x, y:(x,y), vocabulary, count_v))
# sorted by tf
tf_sort    = sorted(term_freq, key = lambda x: x[1], reverse = True)

# get 20 most frequent terms
print("Top 20 frequent terms:")
for i in range(0,19):
    print(tf_sort[i])

'''
The 20 most frequent terms are:
('said', 2180)
('trade', 2154)
('u.s.', 1321)
('billion', 1082)
('dlr', 797)
('japan', 771)
('year', 689)
('export', 634)
('would', 608)
('wa', 600)
('import', 551)
('pct', 526)
('deficit', 524)
('mln', 504)
('market', 448)
('offici', 447)
('surplu', 431)
('ha', 428)
('japanes', 425)
('countri', 421)

The least frequent terms are:
('1.5', 20)
('abov', 20)
('adopt', 20)
('avail', 20)
('carney', 20)
('danger', 20)
('draft', 20)
('exist', 20)
('formal', 20)
('friday', 20)
('gave', 20)
('half', 20)
('immedi', 20)
('lost', 20)
('object', 20)
('push', 20)
('relief', 20)
('request', 20)
('rha', 20)
('risk', 20)
('settl', 20)
('shultz', 20)
('signific', 20)
('similar', 20)
('solut', 20)
('soybean', 20)
('substanti', 20)
('understand', 20)
('unless', 20)
I ignore the terms with frequency less then 20 in this case
'''
# get query list
query_list = ['america export', 'japan import', 'electronic market', 'oil price', 'japan trade war']

# use tf-idf vector space
# get score of query and docs
def similarity(query):
    count_query = vectorizer.transform(query)
    #print(count_query.toarray())

    transformer = TfidfTransformer() 
  # TfidfTransformer takes the CountVectorizer output and computes the tf-idf
    tf_idf      = transformer.fit_transform(count_vect)
    #print(tf_idf.toarray())
    
    similarity  = count_query.dot(tf_idf.transpose())  # dot product 
    array_similarity = similarity.toarray()
    index_order = np.argsort(array_similarity)  # from smallest to largest
    sim_order   = array_similarity[0,index_order[0,::-1]]
    doc_order   = index_order[0,::-1]
    print('\nOrdered similarity: ',sim_order,'\n')
    print('Ordered doc index:  ', doc_order)


for t in query_list:
    similarity([t])
'''
For query "america export":
    top 5 are: [32 434 319 90 232]
    scores:    [0.26855295 0.2526794  0.22976659 0.22415874 0.22326508]
    relavent:  [0 1 0 0 1]
    precision: 0.4
    
    query "japan import":
    top 5 are: [281 2 336 340 152]
    scores:    [0.44652162 0.41587769 0.40873438 0.37558987 0.35495713]
    relavent:  [1 1 1 0 0]
    precision: 0.6
    
    query "electronic market":
    top 5 are: [451 436 177 181 168]
    scores:    [0.46683624 0.25791484 0.22872321 0.20635748 0.19685654]
    relavent:  [1 1 0 1 0]
    precision: 0.6
    
    query "oil price":
    top 5 are: [418 410 360 304 413]
    scores:    [0.37049519 0.30117135 0.29915457 0.29057862 0.28835558]
    relavent:  [0 1 1 1 0]
    precision: 0.6
    
    query "japan trade war":
    top 5 are: [174 164 152  50 309]
    scores:    [0.66953281 0.5885942  0.47316611 0.46888204 0.4631522]
    relavent:  [1 1 1 1 0]
    precision: 0.8
'''
# Assume a = 1, b = 0.2, r = 0.2
# for query "oil price"
#query_mod = count_query + 0.2*count_vect(relavent_index).sum(axis = 0)
count_query_o = vectorizer.transform(['electronic market'])

rel_index     = np.array([451,436,181])
unrel_index   = np.array([177,168])

query_mod     = 0.8*count_query_o + 0.5*count_vect[rel_index,:].sum(axis = 0) - 0.2*count_vect[unrel_index,:].sum(axis = 0)

transformer = TfidfTransformer() 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf      = transformer.fit_transform(count_vect)
    
sim_mod     = query_mod @ tf_idf.transpose()
#array_similarity = sim_mod.toarray()
index_order = np.argsort(sim_mod)  # from smallest to largest
#sim_order   = array_similarity[0,index_order[0,::-1]]
doc_order   = index_order[0,::-1]
print('\nOrdered similarity: ',sim_mod,'\n')
print('Ordered doc index:  ', doc_order)

'''
I'm not sure the syntax of summing specific rows in term-doc matrix.
However, I think the improved query with feed back will be more precise since
it adds more bounds to the query. Additionally, it implies some hidden relation
between the query and desired documents.
'''

'''
Revised:
    For query "oil price":
    Original:
        top 5 are: [418 410 360 304 413]
        relavent:  [0 1 1 1 0]
        precision: 0.6
        
    Modified:
        a = 0.8, b = 0.2, r = 0.2
        top 5 are: [410 360 400 304  79]
        relevant:  [1 1 1 1 0]
        precision: 0.8
        
        a = 0.8, b = 0.5, r = 0.2
        top 5 are: [410 360 400 258 304]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
    For query "america export":
    Original:
        top 5 are: [32 434 319 90 232]
        relavent:  [0 1 0 0 1]
        precision: 0.4 
        
    Modified:
        a = 0.8, b = 0.2, r = 0.2
        top 5 are: [434 232 358 195 293]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
        a = 0.8, b = 0.5, r = 0.2
        top 5 are: [434 232 358 295 243]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
    For query "japan import":
    Original:
        top 5 are: [281 2 336 340 152]
        relavent:  [1 1 1 0 0]
        precision: 0.6
        
    Modified:
        a = 0.8, b = 0.2, r = 0.2
        top 5 are: [2 281  41  49  46]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
        (file41 = file49 but their scores are different
        41 = 6.59364960e+00  49 = 8.97241087e-01)
        
        a = 0.8, b = 0.5, r = 0.2
        top 5 are: [2 281 336   0  49  41] (file49 = file41)
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
    For query "japan trade war":
    Original:
        top 5 are: [174 164 152  50 309]
        relavent:  [1 1 1 1 0]
        precision: 0.8
        
    Modified:
        a = 0.8, b = 0.2, r = 0.2
        top 5 are: [50 152 174  53   7]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
        a = 0.8, b = 0.5, r = 0.2
        top 5 are: [50 152  53 174   7]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
    For query "electronic market":
    Original:
        top 5 are: [451 436 177 181 168]
        relavent:  [1 1 0 1 0]
        precision: 0.6
        
    Modified:
        a = 0.8, b = 0.2, r = 0.2
        top 5 are: [436 451 181 118 121]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
        a = 0.8, b = 0.5, r = 0.2
        top 5 are: [436 451 118 121 181]
        relevant:  [1 1 1 1 1]
        precision: 1.0
        
As we can see, the performance of modified query using feedback methods is 
unbelievably great compared with the original query. 

For low score queries whose precision < 0.6, such as "oil price" and "america export",
when "a" becomes greater, the result becomse better. The reason of this is that since the 
original query doesn't perform well, it needs more feedback modification than well-performed
queries. In this case, the modified part of the new query might dominate the result.

Foe high score queries whose precision >= 0.8, such as "Japan trade war", the change of "a" 
may change the rank order of top results, making new results more close to modified part, 
which might "overfit" in this case. 
Conjecture: When "a" becomes larger, the modified query may not perform well in a different corpus.
'''
print(doc_list[121])