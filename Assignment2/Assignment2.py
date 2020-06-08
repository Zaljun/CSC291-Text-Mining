#Assignment 2
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

# method 2 BM25
def bm25(query, doc_id, doc_freq, term_freq, nr_docs, k):
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
                tf_doc = (k+1) * tf / (k + tf)
                idf    = log( (nr_docs + 1) / doc_freq[t])
                similarity_q_d += tf_query[t] * tf_doc * idf

    return similarity_q_d

# load corpus and generate list of docs 
c_list = reuters.fileids(['grain','corn'])
list_doc = [reuters.raw(t) for t in c_list]

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

#query = 'American corn'
#query = 'government grain'
query = 'American corn market price '
similarity_list = [0 for i in range(nr_docs)]
for doc_id in range(nr_docs):
    similarity_list[doc_id] = tf_idf(query, doc_id, doc_freq, \
                                    term_freq_dict, nr_docs)
# choose k = 2
k = 2
similarity_list_bm25 = [0 for i in range(nr_docs)]
for doc_id in range(nr_docs):
    similarity_list_bm25[doc_id] = bm25(query, doc_id, doc_freq, \
                                    term_freq_dict, nr_docs, k)

similarity_list_id = []
id = 0
for t in similarity_list:
    similarity_list_id.append((id,t))
    id += 1

similarity_list_bm25_id = []
id_bm25 = 0
for t in similarity_list_bm25:
    similarity_list_bm25_id.append((id_bm25,t))
    id_bm25 += 1

#print('sublinear: ',sorted(similarity_list_id, key = lambda x: x[1] ,reverse = True))
#print('\nbm25, k = 2: ',sorted(similarity_list_bm25_id, key = lambda x: x[1], reverse = True))

#print('\nid438: ',reuters.raw('training/4988'))
#print('\nid486: ',reuters.raw('training/6269'))

#print('\nid47: ',reuters.raw('test/15875'))
#print('\nid492: ',reuters.raw('training/6588'))
'''
Sample:
  Query = 'American corn'
  top 5 in sublinear are: (438, 5.9770127468011935), (486, 5.972409146202841), 
                            (369, 5.6254402837002155), (498, 5.174419280344463), 
                            (370, 4.649983519371645)
  top 5 in bm25 (k=2) are: (486, 7.869854153863408), (438, 6.840945777431134), 
                             (369, 6.72582017521578), (498, 6.718598131709877), 
                             (370, 6.199531707945594)
results are different and the different two files are below:
   ------------------------------------------------------------------------------------- 
    
  id438:  NUMEROUS FACTORS SAID POINT TO USSR CORN BUYING
  A greater than anticipated need,
  competitive prices and political motivations could be sparking
  Soviet interest in U.S. corn, industry and government officials
  said.
      As rumors circulated through grain markets today that the
  Soviet Union has purchased an additional 1.5 mln tonnes of U.S.
  corn, industry and government sources noted a number of factors
  that make Soviet buying of U.S. corn likely.
      First, there are supply concerns. Some trade sources said
  recent speculation has been that last year's Soviet grain crop
  be revised to only 190 mln tonnes, rather than the 210 mln
  announced, therby increasing the Soviet need for grain.
      A drop in Argentine corn crop prospects could also affect
  Soviet corn buying, an Agriculture Department source said.
      Dry weather in Argentina -- a major corn supplier to the
  USSR -- and reported crop problems prompted USDA to lower its
  Argentine 1986/87 corn crop estimate this week to 11.0 mln
  tonnes, down from 11.5 mln. Argentina corn exports were also
  cut by 500,000 tonnes to 6.8 mln tonnes.
      Argentina has already committed four mln tonnes of this
  year's corn for export, a USDA official said, with two mln
  tonnes of that booked for April-June delivery to the USSR.
      "Significant downside potential" still exists for the
  Argentine crop, the official said, which will decrease the
  amount of additional corn that country can sell to Moscow.
      "If the Soviet needs are greater than we have been
  thinking, then they might need more than what Argentina can
  provide during the April to June period," he said.
      Current competitive prices for U.S. corn have also sparked
  Soviet buying.
      U.S. corn was reported to be selling on the world market
  earlier this week for around 71 dlrs per tonne, Argentine corn
  for 67 dlrs -- a very competitive price spread, U.S. and Soviet
  sources said.
      "This price difference makes American corn competitive,"
  Albert Melnikov, commercial counselor for the Soviet Union,
  told Reuters.
      Impending crop problems in Argentina will likely cause
  those prices to rise, and with the recently strong U.S. corn
  futures prices, the Soviets might feel corn prices have
  bottomed and that this is a good time to buy, sources said.
      Finally, some industry sources said that by buying the
  minimum amount of corn guaranteed under the U.S./USSR grains
  agreement (four mln tonnes), the Soviet Union may be hoping to
  convince the USDA to offer Moscow a subsidy on wheat.
      In an inteview with Reuters this week, USDA secretary
  Richard Lyng said that no decision had been made on a wheat
  subsidy offer, but that such an offer had not been ruled out.
 
  id486:  MAJOR U.S. FARM GROUPS OPPOSE POLICY CHANGES
  Seven major U.S. farm groups took
  the unusual step of releasing a joint statement urging
  congressional leaders not to tinker with existing farm law.
      Following meetings with House Agriculture Committee
  Chairman Kika de la Garza (D-Tex.) and Senate Agriculture
  Committee Chairman Patrick Leahy (D-Vt.), the groups issued a
  statement saying lawmakers should "resist efforts to overhaul
  the 15-month-old law, which is operating in its first crop
  marketing year."
      The farm groups included the American Farm Bureau
  Federation, American Soybean Association, National Cattlemen's
  Association, National Corn Growers Association, National Cotton
  Council, National Pork Producers Council and the U.S. Rice
  Producers Legislative Group.
      The statement said Congress should not modify the 1985 farm
  bill "so the law might have its intended impact of making
  agriculture more competitive in export markets while at the
  same time maintaining farm income."
      "We strongly believe American farmers now need
  predictability and certainty in farm legislation in order to
  have any opportunity of making proper production and marketing
  decisions," the groups said.  
 ------------------------------------------------------------------------------------- 
 
The reason to this I think is that the query is a little bit common so that 
the results are "ambiguous". Both results are tightly linked to the query.
So I change query to "American corn market price"
New Output:
    top 6 in sublinear are: (438, 10.380114393970572), (498, 9.315376013235282), 
                            (492, 9.112428066245252), (47, 8.84188005923824), 
                            (369, 7.87040998880772),(486, 7.868396173986156)
    top 6 in bm25 (k=2) are: (438, 12.147159667253721), (498, 11.970152315647955), 
                             (486, 10.331652128178924), (492, 9.632241164627699),
                             (369, 9.515576384838344), (47, 8.84188005923824)
The new top two are the same while 3rd, 4th and 6th are different.
Compared with previous query, similarity outputs are higher since the new query 
becomes more specific.
Let's see file 47 and file 492:
    ------------------------------------------------------------------------------------- 
   
  id47:  U.S. SUPPLY/DEMAND DETAILED BY USDA
  The U.S. Agriculture Department made
  the following supply/demand projections for the 1986/87
  seasons, in mln bushels, with comparisons, unless noted --
      CORN --     1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Acreage (mln acres) --
       Planted   76.7    76.7     83.4    83.4
     Harvested   69.2    69.2     75.2    75.2
    Yield (bu)  119.3   119.3    118.0   118.0
   Supply (mln bu) -- 
   Start Stock  4,040   4,040    1,648   1,648
    Production  8,253   8,253    8,877   8,877
       Total-X 12,295  12,295   10,536  10,536
    X-Includes imports.
      CORN (cont.)
                   1986/87           1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Usage: Feed  4,500    4,300     4,095    4,126
         Other  1,180    1,150     1,160    1,129
    Ttl Domest  5,680    5,450     5,255    5,255
       Exports  1,375    1,250     1,241    1,241
     Total Use  7,055    6,700     6,496    6,496
    End Stocks  5,240    5,595     4,040    4,040
   Farmer Reser 1,400    1,300       564      564
    CCC Stocks  1,700    1,500       546      546
   Free Stocks  2,140    2,795     2,930    2,930
    AvgPrice  1.35-1.65  1.35-1.65  2.23     2.23
    Note - Price in dlrs per bu. Corn season begins Sept 1.
      ALL WHEAT -
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Acreage (mln acres) --
       Planted   72.0    72.0     75.6    75.6
     Harvested   60.7    60.7     64.7    64.7
         Yield   34.4    34.4     37.5    37.5
   Supply (mln bu) --
   Start Stcks  1,905   1,905    1,425   1,425
    Production  2,087   2,087    2,425   2,425
         Total
      Supply-X  4,007   4,007    3,865   3,865
   X - Includes imports.
      ALL WHEAT   1986/87            1985/86
   (cont.)   04/09/87 03/09/87  04/09/87 03/09/87
   Usage: Food    700    690       678    678
          Seed     84     90        93     93
          Feed    350    325       274    274
    Ttl Domest  1,134  1,105     1,045  1,045
       Exports  1,025  1,025       915    915
     Total Use  2,159  2,130     1,960  1,960
    End Stocks  1,848  1,877     1,905  1,905
   Farmer Reser   475    450       433    433
    CCC Stocks    950    950       602    602
   Free Stocks    423    477       870    870
    Avg Price  2.30-40  2.30-40   3.08   3.08
   Note - Price in dlrs per bushel. Wheat season begins June 1.
      SOYBEANS -
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Acreage (mln acres) --
        Planted   61.5   61.5     63.1   61.1
      Harvested   59.4   59.4     61.6   61.6
     Yield (bu)   33.8   33.8     34.1   34.1
   Supply (mln bu) --
   Start Stocks    536    536      316    316
     Production  2,007  2,007    2,099  2,099
          Total  2,543  2,543    2,415  2,415
      SOYBEANS (cont.)
                 1986/87             1985/86
            04/09/87 03/09/87   04/09/87 03/09/87
        Usage --
    Crushings  1,130   1,115      1,053   1,053
      Exports    700     700        740     740
   Seed, Feed and
     Residual    103      93         86      86
    Total Use  1,933   1,908      1,879   1,879
   End Stocks    610     635        536     536
   Avg Price 4.60-4.80 4.60-4.80   5.05    5.05
   Note - Average price in dlrs per bushel. Soybean season begins
  June 1.
      FEEDGRAINS - X
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Acreage (mln acres) --
        Planted  119.8   119.8     128.1   128.1
      Harvested  102.0   102.0     111.8   111.8
   Yld (tonnes)   2.48    2.48      2.45    2.45
   Supply (mln tonnes) --
   Start Stocks  126.4   126.4      57.5    57.5
     Production  252.4   252.4     274.4   274.4
        Imports    0.6     0.6       0.9     0.9
          Total  379.4   379.4     332.7   332.7
   X - Includes corn, sorghum, barley, oats.
      FEEDGRAINS - X (cont.)
                   1986/87          1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Usage: Feed  140.6   136.2     134.8   135.5
         Other   35.8    35.0      35.0    34.3
    Ttl Domest  176.4   171.2     169.8   169.8
       Exports   43.9    40.8      36.6    36.6
     Total Use  220.3   211.9     206.4   206.4
    End Stocks  159.1   167.5     126.4   126.4
   Farmer Reser  39.0    36.5      16.6    16.6
    CCC Stocks   55.2    49.5      20.4    20.4
   Free Stocks   64.8    81.5      89.3    89.3
   X - Includes corn, sorghum, oats, barley. Seasons for oats,
  barley began June 1, corn and sorghum Sept 1.
      SOYBEAN OIL -
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Supply (mln lbs) --
   Start Stcks     947     947       632     632
    Production  12,263  12,103    11,617  11,617
       Imports     Nil     Nil         8       8
         Total  13,210  13,050    12,257  12,257
   Note - 1985/86 production estimates based on October year
  crush of 1,060 mln bushels.
      SOYBEAN OIL (cont.) -
                 1986/87             1985/86
            04/09/87  03/09/87  04/09/87  03/09/87
   Usage (mln lbs) --
    Domestic  10,500    10,500    10,053   10,053
     Exports   1,350     1,350     1,257    1,257
       Total  11,850    11,850    11,310   11,310
   End Stcks   1,360     1,200       947      947
   AvgPrice  14.5-16.0  15.0-17.0  18.00    18.00
   Note - Average price in cents per lb. Season for soybean oil
  begins Oct 1.
      SOYBEAN CAKE/MEAL, in thousand short tons --
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Start Stcks     212     212       387     387
    Production  26,558  26,203    24,951  24,951
         Total  26,770  26,415    25,338  25,338
   Note - 1985/86 production estimates based on October year
  crush of 1,060 mln bushels.
      SOY CAKE/MEAL (cont.) -
                 1986/87            1985/86
            04/09/87 03/09/87  04/09/87 03/09/87
   Usage (thous short tons) --
    Domestic   20,000  19,750    19,090  19,118
     Exports    6,500   6,350     6,036   6,008
       Total   26,500  26,100    25,126  25,126
   End Stcks      270     315       212     212
    AvgPrice  145-150  145-150   154.90  154.90
    Note - Price in dlrs per short ton. Season for soybean cake
  and meal begins Oct 1.
      COTTON --
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Area (mln acres) --
        Planted  10.06   10.06     10.68   10.68
      Harvested   8.49    8.49     10.23   10.23
    Yield (lbs)    549     553       630     630
   Supply (mln 480-lb bales) --
   Start Stks-X   9.35    9.35      4.10    4.10
     Production   9.70    9.79     13.43   13.43
   Ttl Supply-Y  19.06   19.14     17.57   17.57
    X - Based on Census Bureau data. Y - Includes imports.
      COTTON (cont.) -
                 1986/87           1985/86
            04/09/87 03/09/87  04/09/87 03/09/87
     Usage --
     Domestic   7.10    7.01     6.40    6.40
      Exports   6.66    6.76     1.96    1.96
        Total  13.76   13.77     8.36    8.36
   End Stocks   5.40    5.49     9.35    9.35
   Avge Price  51.7-X  51.7-X   56.50   56.50
   X - 1986/87 price is weighted average for first five months of
  marketing year, not a projection for 1986/87. Average price in
  cents per lb. Cotton season begins August 1.
      RICE
                 1986/87            1985/86
            04/09/87 03/09/87  04/09/87 03/09/87
   Acreage (mln acres) --
       Planted   2.40    2.40      2.51    2.51
     Harvested   2.38    2.38      2.49    2.49
   Yield (lbs)  5,648   5,648     5,414   5,414
   Supply (mln cwts) --
   Start Stcks   77.3    77.3      64.7    64.7
    Production  134.4   134.4     134.9   134.9
       Imports    2.2     2.2       2.2     2.2
         Total  213.9   213.9     201.8   201.8
      RICE (cont.)
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
   Usage (mln cwts) --
      Domestic   67.0    67.0      65.8    65.8
       Exports   80.0    80.0      58.7    58.7
       Total-Y  147.0   147.0     124.5   124.5
    End Stocks   66.9    66.9      77.3    77.3
    CCC Stocks   42.9    42.9      41.5    41.5
   Free Stocks   24.0    24.0      35.8    35.8
     AvgPrice 3.45-4.25 3.45-4.25  6.53    6.53
   Note - Average price in dlrs per CWT. Y-Rough equivalent.
  N.A.-Not Available, USDA revising price definition due to
  marketing loan. Rice season begins August 1.
      SORGHUM
                 1986/87            1985/86
            04/09/87 03/09/87  04/09/87 03/09/87
    Yield (bu)  67.7    67.7     66.8    66.8
   Supply (mln bu) --
   Start Stcks   551     551      300     300
    Production   942     942    1,120   1,120
         Total 1,493   1,493    1,420   1,420
    Usage (mln bu) --
          Feed   550     575      662     662
         Other    30      30       29      29
    Ttl Domest   580     605      691     691
      SORGHUM (cont.) -
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
      Exports     225     225       178     178
    Total Use     805     830       869     869
   End Stocks     688     663       551     551
   Avge Price  1.30-50  1.30-50    1.93    1.93
   Note - Price in dlrs per bushel. Sorghum season begins Sept 1.
      BARLEY
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
     Yield (bu)  50.8    50.8      51.0    51.0
   Start Stocks   325     325       247     247
     Production   610     610       591     591
        Imports     5       5         9       9
          Total   941     941       847     847
      BARLEY (cont.)
               1986/87            1985/86
          04/09/87 03/15/87  04/09/87 03/15/87
   Usage (mln bu) --
         Feed  300    300       333    333
        Other  175    175       167    167
   Ttl Domest  475    475       500    500
      Exports  150    150        22     22
    Total Use  625    625       522    522
   End Stocks  316    316       325    325
    AvgPrice 1.45-65  1.45-65  1.98   1.98
   Note - Average price in dlrs per bushel. Barley season begins
  June 1.
      OATS - in mln bushels
                 1986/87            1985/86
            04/09/87 03/09/87  04/09/87 03/09/87
    Yield (bu)  56.0   56.0      63.7   63.7
   Start Stcks   184    184       180    180
    Production   385    385       521    521
       Imports    30     30        28     28
         Total   598    598       729    729
      OATS, in mln bushels (cont.)
                1986/87             1985/86
           04/09/87 03/09/87   04/09/87 03/09/87
       Usage --
        Feed   400     400       460     460
       Other    85      85        83      83
   Ttl Domes   485     485       543     543
     Exports     2       2         2       2
       Total   487     487       545     545
   End Stcks   111     111       184     184
   AvgPrice 1.00-20  1.00-20    1.23    1.23
    Note - Average price in dlrs per bushel. Oats season begins
  June 1.
      LONG GRAIN RICE, in mln CWTs (100 lbs) --
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
     Harvested --
   Acres (mln)   1.83    1.83     1.94    1.94
   Yield (lbs)  5,358   5,358    5,168   5,168
    Start Stks   49.3    49.3     37.7    37.7
    Production   97.8    97.8    100.4   100.4
    Ttl Supply  148.6   148.6    140.1   140.1
   Note -- Starting Stocks does not include broken kernels --
  Supply minus use does not equal ending stocks in breakdowns.
  Total Supply includes imports but not broken kernels.
      LONG GRAIN RICE, in mln CWTs (100 lbs), cont. --
                   1986/87           1985/86
              04/09/87 03/09/87 04/09/87 03/09/87
   Domestic Use   43.0    43.0      48.8    48.8
        Exports   65.0    60.0      42.0    42.0
      Total Use  108.0   103.0      90.8    90.8
   End Stocks-X   40.6    45.6      49.3    49.3
      AvgPric 3.45-4.25 3.45-4.24   6.86    6.86
   Note - Average price in dlrs per cwt. X-Broken kernels not
  included -- supply minus use does not equal ending stocks in
  breakdowns. Rice season begins August 1.
      MEDIUM, SHORT GRAIN RICE - in mln CWTs (100 lbs) --
                  1986/87            1985/86
             04/09/87 03/09/87  04/09/87 03/09/87
     Harvested --
   Acres (mln)   0.55    0.55     0.55    0.55
   Yield (lbs)  6,651   6,651    6,258   6,258
    Start Stks   26.7    26.7     25.7    25.7
    Production   36.6    36.6     34.5    34.5
    Ttl Supply   65.3    65.3     61.7    61.7
   Note -- Starting Stocks does not include broken kernels --
  Supply minus use does not equal ending stocks in breakdowns.
  Total Supply includes imports but not broken kernels.
      MEDIUM, SHORT GRAIN RICE, in mln CWTs (100 lbs), cont. --
                   1986/87          1985/86
              04/09/87 03/09/87 04/09/87 03/09/87
   Domestic Use   24.0    24.0      17.0    17.0
        Exports   15.0    20.0      16.7    16.7
      Total Use   39.0    44.0      33.7    33.7
   End Stocks-X   24.5    19.5      26.7    26.7
      AvgPric 3.45-4.25  3.45-4.25  5.91    5.91
   Note - Average price in dlrs per CWT. X-Broken kernels not
  included - supply minus use does not equal ending stocks in
  breakdowns. Rice season begins August 1.
      NOTES ON U.S. SUPPLY/DEMAND TABLES
      -- N.A. - Not available.
      -- Totals may not add due to rounding.
      -- Figures for 1986/87 are midpoint of USDA range.
      -- Feed usage for corn, wheat, soybean, feedgrains,
  sorghum, barley, oats includes residual amount.
      -- Residual amount included in rice and medium/short grain
  rice domestic usage.
      -- Rice, long grain, and medium/short grain rice average
  price for 1985/86 estimates and 1986/87 projections are market
  prices and exclude cash retained under the marketing loan since
  April, 1986.
  
  id492:  U.S. CORN MARKET SKEWED BY SOVIET BUYING
  Recent purchases of U.S. corn by the
  Soviet Union have skewed the domestic cash market by increasing
  the price difference between the premium price paid at the Gulf
  export point and interior levels, cash grain dealers said.
      Many dealers expect the USDA will act soon to reduce the
  cash price premium at the Gulf versus the interior -- which a
  dealer in Davenport, Iowa, said was roughly 20 pct wider than
  normal for this time of year at 25 cents a bushel -- by making
  it worthwhile for farmers to move grain.
      By lowering ASCS county posted prices for corn, the USDA
  could encourage farmers to engage in PIK and roll corn sales,
  where PIK certificates are used to redeem corn stored under the
  government price support loan program and then marketed.
      If the USDA acts soon, as many dealers expect, the movement
  would break the Gulf corn basis.
      "The USDA has been using the Gulf price to determine county
  posted prices," one dealer said. "It should be taking the
  average of the Gulf price and the price in Kansas City," which
  would more closely reflect the lower prices in the interior
  Midwest.
      "But we don't know when they might do it," an Ohio dealer
  said, which has created uncertainty in the market.
      The USDA started the PIK certificate program in an effort
  to free up surplus grain that otherwise would be forfeited to
  the government and remain off the market and in storage.
      Yesterday, USDA issued a report showing that only slightly
  more than 50 pct of the 3.85 billion dlrs in PIK certificates
  it has issued to farmers (in lieu of cash payments) had to date
  been exchanged for grain.
      With several billion dlrs worth of additional PIK
  certificates scheduled to be issued in the coming months, the
  USDA would be well advised to encourage the exchange for grain
  by adjusting the ASCS prices, cash grain dealers said.
      A byproduct of the Soviet buying has been a sharp rise in
  barge freight costs quoted for carrying grain from the Midwest
  to the export terminals, cash dealers said.
      Freight from upper areas of the Mississippi have risen
  nearly 50 pct in the past two weeks to over 150 pct of the
  original tariff price. The mild winter and early reopening of
  the mid-Mississippi river this spring have also encouraged the
  firmer trend in barge freight, dealers noted.
      The higher transportation costs have served to depress
  interior corn basis levels, squeezing the margins obtained by
  the elevators feeding the Gulf export market as well as
  discouraging farmer marketings, they said.
      "The Gulf market overreacted to the Soviet buying reports,"
  which indicate the USSR has booked over two and perhaps as much
  as 4.0 mln tonnes of U.S. corn, one Midwest cash grain trader
  said.
      But dealers anticipate that once the rumors subside,
  freight rates will settle back down because of the overall
  surplus of barges on the Midwest river system.
  ------------------------------------------------------------------------------------- 
As we can see, file 47 is a massive long record of corn price in US,
which repeats terms in query many times, when ignoring the numbers 
in file.
I think file 47 is better when you are serching a history record of 
corn market price, while file 492 is better when searching for a news.
Maybe we can add "Tag" to files to indicate its "type" to make result 
more precious when the similarities are close.
'''