# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 19:58:02 2019

@author: admin
"""

from nltk.corpus import reuters

fid = reuters.fileids()  # returns a list of training/test file ids
categories = reuters.categories()   # returns a list of all categories 
categories_fid = reuters.categories(fid[10])  # returns the categories of a text 
reuters.raw(fid[10])    # raw text 
reuters.words(fid[10])  # parsed into words
#c_list = reuters.fileids(['grain','corn','cotton'])
c_list = reuters.fileids('trade')
doc_list = [reuters.raw(t) for t in c_list]
#for i in range(9):
 #   text += reuters.raw(fid[i])
#print(categories)
#print(doc_list)
print(reuters.raw(['training/9784','training/9848']))