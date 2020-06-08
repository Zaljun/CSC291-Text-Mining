# list of documents
# Freq_Dist 
#  from nltk.probability import FreqDist
#  tokens = nltk.word_tokenize(sentence)
#  fdist=FreqDist(tokens)  --> returns the frequency distribution of words

list_doc =[ "Information retrieval course will cover techniques used in search engines for finding relevant documents or information related to a query. Topics include: natural language processing for extracting relevant terms out of text data, vector space a methods for computing similarity between documents, text classification, and clustering. These techniques are commonly used in applications such as: automatic extraction of summaries out of a long text, extract novel information in a stream of data.",  

"NLP algorithms are typically based on machine learning algorithms. Instead of hand-coding large sets of rules, NLP can rely on machine learning to automatically learn these rules by analyzing a set of examples (i.e. a large corpus, like a book, down to a collection of sentences), and making a statical inference. In general, the more data analyzed, the more accurate the model will be.",  

"The Denver Broncos made sure Brandon McManus will be their kicker for the long haul on Monday.  General manager John Elway announced the team and the kicker agreed on a contract extension. NFL Network Insider Ian Rapoport reported, per a source, it's a three-year extension worth $11.254 million with $6 million of it guaranteed. McManus is now the NFL's fourth highest paid kicker.", 

 "Equifax, one of the three major credit reporting agencies, handles the data of 820 million consumers and more than 91 million businesses worldwide. Between May and July of this year 143 million people in the U.S. may have had their names, Social Security numbers, birth dates, addresses and even driver's license numbers accessed. In addition, the hack compromised 209,000 people's credit card numbers and personal dispute details for another 182,000 people. What bad actors could do with that information is daunting. This data breach is more confusing than others -- like when Yahoo or Target were hacked, for example -- according to Joel Winston, a former deputy attorney general for New Jersey , whose current law practice focuses on consumer rights litigation, information privacy, and data protection law.", 
 
  """Why didn't she text me back yet? She doesn't like me anymore!" "There's no way I'm trying out for the team. I suck at basketball""It's not fair that I have a curfew! "Sound familiar? Parents of tweens and teens often shrug off such anxious and gloomy thinking as normal irritability and moodiness — because it is. Still, the beginning of a new school year, with all of the required adjustments, is a good time to consider just how closely the habit of negative, exaggerated "self-talk" can affect academic and social success, self-esteem and happiness. Psychological research shows that what we think can have a powerful influence on how we feel emotionally and physically, and on how we behave. Research also shows that our harmful thinking patterns can be changed."""]

  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
import numpy as np
porter = PorterStemmer()

import spacy   # another tokenizer, lemmatizer
spacy.load('en_core_web_sm')  

def nlp_processing(doc):
    stoplist = stopwords.words('english') 
    lemmatizer = spacy.lang.en.English()  
    tokens = lemmatizer(doc)
    #print(type(tokens))
    terms_no_stop = [token.lemma_ for token in tokens if token.lemma_ not in stoplist]
    terms_alnum   = [t for t in terms_no_stop if t.isalpha()]
    return terms_alnum

vectorizer = CountVectorizer(tokenizer = nlp_processing) # min_df = 2) 
# stop_words = 'english', max_df, min_df, max_features, try analyzer = 'word'
count_vect = vectorizer.fit_transform(list_doc)  
# type count_vect= scipy sparse matrix
#print(vectorizer.get_feature_names())  # prints the terms (vocabulary)
#print(count_vect.toarray())       # converts it to a regular scipy matrix
#print(count_vect.shape)           # prints the shape 

# query vector 
query = ["natural language processing"]
count_query = vectorizer.transform(query)
print(count_query.toarray())

transformer = TfidfTransformer(norm = None, sublinear_tf = True) 
# TfidfTransformer takes the CountVectorizer output and computes the tf-idf
tf_idf = transformer.fit_transform(count_vect)
print(tf_idf.toarray())  # convert to a non-sparse (regular matrix)
#, min_df = 2, 
# TfidfVectorizer = does the same thing as CountVectorizer + TfidfTransformer
# tf_idf_vect = TfidfVectorizer(tokenizer = nlp_processing,sublinear_tf = True) #, min_df = 2, 
# Other parameters: stop_words = 'english', max_df, min_df, max_features

similarity       = count_query.dot(tf_idf.transpose())  # dot product 
array_similarity = similarity.toarray()
index_order      = np.argsort(array_similarity)  # from smallest to largest

print('Doc index: ', index_order, index_order.shape)
print('Query similarity: ', array_similarity)

print('Ordered similarity: ',array_similarity[0,index_order[0,::-1]])
print('Ordered doc index:  ', index_order[0,::-1])

index_rel = np.array([0,1])
index_irel = np.array(4)
#query_modified = count_query + 0.2 * tf_idf[0] + 0.2*tf_idf[1] - 0.2*tf_idf[4]
query_modified = count_query + 0.2 * tf_idf[index_rel,:].sum( axis = 0) - 0.2*tf_idf[index_irel,:].sum(axis = 0)


similarity_qm    = query_modified @ tf_idf.transpose()
#array_similarity = similarity_qm.toarray()
index_order      = np.argsort(similarity_qm)  # from smallest to largest

print('Doc index: ', index_order, index_order.shape)
print('Query similarity: ', similarity_qm)

print('Ordered similarity: ',similarity_qm[0,index_order[0,::-1]])
print('Ordered doc index:  ', index_order[0,::-1])
#print('\tDoc', '\tSimilarity')
#for  in :
#    print('\t', doc, '\t', sim)