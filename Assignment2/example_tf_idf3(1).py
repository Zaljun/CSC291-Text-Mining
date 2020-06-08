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

from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from collections import Counter
from math import log

# Step 0: preprocess each document

# input a string of characters
# output a list of tokens
def process(text):
    all_tokens   = word_tokenize(text)   # tokenize text
    all_pos_tags = pos_tag(all_tokens) # adds POS to each token
    
    # 1. eliminate terms with POS with a length = 1 (all tags equal to '.')
    no_punkt_pos = [(t, pos) for (t,pos) in all_pos_tags if len(pos) > 1]  
    # 2. convert to lower case
    lower_tokens = [(t.lower(),pos) for (t,pos) in no_punkt_pos]
    # 3. stem the words
    #porter = PorterStemmer()
    #stemmed_tokens = [(porter.stem(t),pos) for (t, pos) in lower_tokens]
    stemmed_tokens = lower_tokens
    
    # eliminate stop words
    stoplist = stopwords.words('english') # from nltk
    stoplist.extend(["n't", "'s"])
    no_stopwords = [(t, pos) for (t, pos) in stemmed_tokens if t not in stoplist]
    
    # eliminated POS from the final list
    good_tokens = [ t for (t,pos) in no_stopwords]
    #print(good_tokens)
    
    return good_tokens


# Step 1: extract a list of (token, doc_id) from all documents.
# input a list of documents
# output: a list of (token, doc_id) tuples
def extract_token_doc_id(list_doc):
    
    all_tokens = []
    for doc_id, text in enumerate(list_doc):  # doc_id = 0 to len(list_doc)-1
        good_tokens = process(text)
        this_doc_tokens  = [ (t, doc_id) for t in good_tokens]
        all_tokens.extend(this_doc_tokens)
    
    return all_tokens


def tf_idf(query, doc_id, doc_freq, term_freq, nr_docs):

    # process the query same as a regular document
    good_tokens = process(query)

    # Compute the term frequency for all unique terms in the query
    tf_query = Counter(good_tokens)

    # Compute the similarity value between query and doc_id

    similarity_q_d = 0
    # for all terms in the query
    for t in tf_query:
    #  if that term has a non-zero freq in doc_id -> add it to the sum
        if t in term_freq:
            # tf = term frequency of t in doc_id
            if doc_id in term_freq[t]:
                tf     = term_freq[t][doc_id]
                tf_doc = log(1 + tf)
                idf    = log( (nr_docs + 1) / doc_freq[t])
                similarity_q_d += tf_query[t] * tf_doc * idf

    return similarity_q_d


all_tokens = extract_token_doc_id(list_doc)

# Step 2: Sort by token and doc_id
sorted_all_tokens = sorted(all_tokens)

# Step 3: Extract term_freg and doc_freq

# Extract term frequency
term_freq = Counter(sorted_all_tokens) # dictionary of (term, doc_id): term_freq

# Extract dictionary of {term: [ (doc_id, term_freq), .... ]}
term_freq_dict = {}
for (term,doc_id) in term_freq:
    if term in term_freq_dict:
        term_freq_dict[term][doc_id] = term_freq[(term,doc_id)]
    else:
        term_freq_dict[term] = {doc_id:term_freq[(term,doc_id)]}

# Extract document frequency
tokens_doc = [token for (token, did) in term_freq]	
doc_freq  = {token:tokens_doc.count(token) \
                    for token in set(tokens_doc)}

# Compute tf_idf similarity function
# between a query and a document 
nr_docs = len(list_doc)

query = 'machine learning'
#query = 'data automation'
#query = 'yahoo data'
similarity_list = [0 for i in range(nr_docs)]
for doc_id in range(nr_docs):
    similarity_list[doc_id] = tf_idf(query, doc_id, doc_freq, \
                                    term_freq_dict, nr_docs)
    
print(similarity_list)
max(similarity_list)






