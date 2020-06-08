# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:00:48 2019

@author: admin
"""
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from nltk.corpus import reuters

def process(text):
    all_tokens   = word_tokenize(text)  
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
    stoplist.extend(["n't", "'s"])
    no_stopwords = [(t, pos) for (t, pos) in stemmed_tokens if t not in stoplist]
    
    good_tokens = [ t for (t,pos) in no_stopwords]
 
    return good_tokens

categories = reuters.categories()   # returns a list of all categories 

c_list = reuters.fileids('trade')
doc_list = [reuters.raw(t) for t in c_list]

total = [process(t) for t in doc_list]
count = 0
for t in total:
    count += len(t)
count_average = count/485
#print(average_count)

min = 99999
max = 0
for t in total:
    if len(t)<=min:
        min = len(t)
    if len(t)>=max:
        max = len(t)
#print(total)
#print(categories)
#print(doc_list)

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