#Assignment 3
#Zhaojun Jia


from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords 
from collections import Counter
from math import log
from nltk.corpus import reuters

#text preprocessing
def process(text):
    all_tokens   = word_tokenize(text)   # tokenize text
    all_pos_tags = pos_tag(all_tokens) # adds POS to each token
    
    # eliminate terms with POS with a length = 1 (all tags equal to '.')
    # and do further elimination
    tokens_no_punct = [(t, pos) for (t,pos) in all_pos_tags if len(pos) > 1]  
    no_punkt_pos = []
    for (t,pos) in tokens_no_punct:
        valid = False
        for ch in t:
            if ch.isalnum():
                valid = True
        if valid:
            no_punkt_pos.append((t,pos))
            
    # convert to lower case
    lower_tokens = [(t.lower(),pos) for (t,pos) in no_punkt_pos]
    
    # stem the words
    porter = PorterStemmer()
    stemmed_tokens = [(porter.stem(t),pos) for (t, pos) in lower_tokens]
    
    # eliminate stop words
    stoplist = stopwords.words('english')
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

# method 1
def JM(query, doc_id, term_freq, lam, doc_list):

    # process the query same as a regular document
    good_tokens = process(query)

    # Compute the term frequency for all unique terms in the query
    tf_query = Counter(good_tokens)

    # Compute the similarity value between query and doc_id

    p_q_d = 0
    p_q_c = 0
    # for all terms in the query
    for t in tf_query:
    #  if that term has a non-zero freq in doc_id -> add it to the sum
        if t in term_freq:
            # tf = term frequency of t in doc_id
            if doc_id in term_freq[t]:
#                tf     = term_freq[t][doc_id]
 #               p_w_d  = tf/Counter(doc_list[doc_id])
  #              p_q_d += tf_query[t] * log(p_w_d)
        if t not in term_freq:
#            p_w_d = nr_t / Counter(all_tokens)
 #           p_q_c += log(p_w_d)
        score_q_d = (1 - lam)*p_q_d + lam*p_q_c

    return score_q_d

#def Dirichlet():
    #similar as JM
    #try to use average length as mu

    
    
# load corpus and generate list of docs 
#c_list = reuters.fileids(['grain','corn'])
c_list = reuters.fileids('grain')
list_doc = [reuters.raw(t) for t in c_list]

all_tokens = extract_token_doc_id(list_doc)

# Step 2: Sort by token and doc_id
sorted_all_tokens = sorted(all_tokens, key = lambda x: x[1], reverse = True)

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

#query = 'American corn'
#query = 'government grain'
query = 'American corn market price '
