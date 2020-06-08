# Assignment1_fall
# h702915924
# Zhaojun Jia

from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import PorterStemmer 
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from string import punctuation

def read_line_by_line():
    # read the alice file line by line
    f = open('alice.txt', 'r')
    #print(f.read)
    text = ""
    for line in f:      ## iterates over the lines of the file
        print(line, end = '')    ## trailing , so print does not add an end-of-line char
                   ## since 'line' already includes the end-of line.
        # put read lines into text            
        text += line
    f.close()
    return text

def process(text):

	# word tokenize 
    tokens = word_tokenize(text)
    #print("word tokens: ",tokens)
    
    # eliminate punctuation
    tokens_no_punct = [t for t in tokens if t not in punctuation]
    
    # further eliminate punctuation such as "
    no_punct = []
    for t in tokens_no_punct:
        valid = False
        for ch in t:
            if ch.isalnum():
                valid = True
        if valid:
            no_punct.append(t)
            
    # lower tokens
    lower_tokens    = [t.lower() for t in no_punct]
    #print("\nLower tokens with no ponctuation: ", lower_tokens)
    
    # eliminte stopwords
    stoplist = stopwords.words('english')
    no_stopwords = [t for t in lower_tokens if t not in stoplist]
    #print("\nNo stop words: ", no_stopwords)
    
    # lemmatization
    wnl = WordNetLemmatizer()
    lemmatized_tokens = [wnl.lemmatize(t) for t in no_stopwords]
    #print("\nLemmatized tokens: ",lemmatized_tokens)
    
    # filter POS
    lem_tokens_tag = pos_tag(lemmatized_tokens)
    #print("\nPOS tokens: ",lem_tokens_tag)
    
    # convert tags to 'n' 'v' 'a' 'r'
    # all POS starting with N  -> n (nouns)
    # all POS starting with V  -> v (verbs)
    #                       J  -> a (adjectives)
    #                       R  -> 'r' (adverbs)
    tokens_tag = []
    for tupl in lem_tokens_tag:
        if tupl[1][0] == 'N':
            x = list(tupl)
            #print("list",x)
            x[1] = 'n'
            #print(x)
            tupl = tuple(x)
            #print("Tuple",tupl)
            tokens_tag.append(tupl)
        elif tupl[1][0] == 'V':
            x = list(tupl)
            x[1] = 'v'
            tupl = tuple(x)
            tokens_tag.append(tupl)
        elif tupl[1][0] == 'J':
            x = list(tupl)
            x[1] = 'a'
            tupl = tuple(x)
            tokens_tag.append(tupl)
        elif tupl[1][0] == 'R':
            x = list(tupl)
            x[1] = 'r'
            tupl = tuple(x)
            tokens_tag.append(tupl)
    #this method below doesn't work because tuple won't change its value
    '''for (t,pos) in lem_tokens_tag:
        print((t,pos))
        if pos in 'nvar':
            tokens_tag.append((t,pos))'''
    #print("\nAfter POS filtering: ",tokens_tag)
    
    #count tokens with POS
    unique_tokens = sorted(set(tokens_tag))
    #print("\nUnique tokens: ",unique_tokens)

    freq_tokens = []
    for item in unique_tokens:
        freq_tokens.append((item[0],item[1],tokens_tag.count(item)))
    #print("\nToken frequency: ", freq_tokens)
        
	#count terms without POS
    just_terms = [t for (t,pos) in tokens_tag]
    unique_terms = sorted(set(just_terms))
    #print("\nUnique terms: ",unique_terms)
    
    freq_terms = [(t,just_terms.count(t)) for t in unique_terms]
    #print("\nTerm frequency: ", freq_terms)
    
    #sort by frequency
    sorted_terms = sorted(freq_terms,key = lambda x: x[1], reverse = True)
    print("\nSorted by frequency: ",sorted_terms)
    
    return freq_tokens
    
pos = process(read_line_by_line())

""" Sample:
    Sorted by frequency:  [('said', 462), ('alice', 399), ("n't", 208), ('little', 128), ('know', 89), ('went', 83), ('thing', 79), ('queen', 76), ('thought', 76), ('time', 74), ('see', 67), ('king', 64), ('head', 60), ('turtle', 60), ('well', 60), ('began', 58), ('go', 57), ('hatter', 57), ('mock', 56), ('gryphon', 55), ('quite', 55), ('say', 55), ('way', 54), ('think', 53), ('first', 51), ('much', 51), ('voice', 51), ("'m", 50), ('cat', 50), ('come', 48), ('never', 47), ('rabbit', 47), ('get', 46), ('got', 45), ('looked', 45), ('mouse', 45), ('duchess', 42), ('tone', 42), ("'ve", 40), ('came', 40), ('round', 40), ('back', 39), ('great', 39), ('make', 38), ("'re", 36), ('dormouse', 35), ('eye', 35), ('march', 34), ('nothing', 34), ('oh', 34), ('tell', 34), ('large', 33), ('last', 33), ('dear', 32), ('door', 32), ('found', 32), ('hand', 32), ('long', 32), ('minute', 32), ('hare', 31), ('moment', 31), ('put', 31), ('right', 31), ('word', 31), ('heard', 30), ('looking', 30), ('made', 30), ('white', 30), ('day', 29), ('foot', 29), ('replied', 29), ('next', 28), ('caterpillar', 27), ('going', 27), ('look', 27), ('poor', 27), ('seemed', 27), ('away', 25), ('course', 25), ('rather', 25), ('soon', 25), ('yet', 25), ('good', 24), ('sure', 24), ('take', 24), ('took', 24), ('added', 23), ('felt', 23), ('sort', 23), ('getting', 22), ('ever', 21), ('find', 21), ('half', 21), ('let', 21), ('side', 21), ('wish', 21), ('anything', 20), ('child', 20), ('cried', 20), ('face', 20), ('however', 20), ('question', 20), ('till', 20), ('curious', 19), ('even', 19), ('house', 19), ('old', 19), ('please', 19), ('tried', 19), ('arm', 18), ('court', 18), ('eat', 18), ('end', 18), ('enough', 18), ('something', 18), ('soup', 18), ('table', 18), ('wonder', 18), ('asked', 17), ('begin', 17), ('bill', 17), ('jury', 17), ('perhaps', 17), ('sat', 17), ('spoke', 17), ('talking', 17), ('use', 17), ('bit', 16), ('change', 16), ('garden', 16), ('hastily', 16), ('high', 16), ('indeed', 16), ('ran', 16), ('turned', 16), ('air', 15), ('called', 15), ('done', 15), ('gave', 15), ('idea', 15), ('life', 15), ('mad', 15), ('mean', 15), ('saying', 15), ('seen', 15), ('speak', 15), ('anxiously', 14), ('beginning', 14), ('better', 14), ('certainly', 14), ('creature', 14), ('dinah', 14), ('everything', 14), ('game', 14), ('hear', 14), ('knew', 14), ('left', 14), ('lobster', 14), ('low', 14), ('mouth', 14), ('remember', 14), ('saw', 14), ('set', 14), ('silence', 14), ('size', 14), ('suppose', 14), ('talk', 14), ('trying', 14), ('always', 13), ('beautiful', 13), ('close', 13), ('cook', 13), ('dance', 13), ('dodo', 13), ('far', 13), ('gone', 13), ('grow', 13), ('kept', 13), ('people', 13), ('really', 13), ('remark', 13), ('room', 13), ('sea', 13), ('still', 13), ('suddenly', 13), ('tea', 13), ('turn', 13), ('u', 13), ('used', 13), ('whole', 13), ('afraid', 12), ('baby', 12), ('best', 12), ('bird', 12), ('chapter', 12), ('deal', 12), ('else', 12), ('finished', 12), ('footman', 12), ('give', 12), ('hardly', 12), ('like', 12), ('majesty', 12), ('many', 12), ('name', 12), ('pigeon', 12), ('serpent', 12), ('tail', 12), ('tree', 12), ('try', 12), ('turning', 12), ('ask', 11), ('conversation', 11), ('ear', 11), ('glad', 11), ('glove', 11), ('growing', 11), ('hurried', 11), ('hurry', 11), ('keep', 11), ('lesson', 11), ('matter', 11), ('mind', 11), ('nearly', 11), ('pool', 11), ('read', 11), ('sister', 11), ('slate', 11), ('soldier', 11), ('thinking', 11), ('trial', 11), ('waited', 11), ('want', 11), ('bottle', 10), ('explain', 10), ('fan', 10), ('glass', 10), ('heart', 10), ('hedgehog', 10), ('hold', 10), ('offended', 10), ('opened', 10), ('pig', 10), ('place', 10), ('queer', 10), ('reason', 10), ('remarked', 10), ('repeated', 10), ('rest', 10), ('sight', 10), ('sitting', 10), ('small', 10), ('walked', 10), ('witness', 10), ('angrily', 9), ('answer', 9), ('believe', 9), ('call', 9), ('coming', 9), ('continued', 9), ('different', 9), ('feeling', 9), ('hall', 9), ('help', 9), ("i'm", 9), ('interrupted', 9), ('join', 9), ('key', 9), ('knave', 9), ('least', 9), ('leave', 9), ('mine', 9), ('moral', 9), ('puzzled', 9), ('rate', 9), ('shook', 9), ('shouted', 9), ('tear', 9), ('timidly', 9), ('together', 9), ('top', 9), ('waiting', 9), ('work', 9), ('adventure', 8), ('appeared', 8), ('asleep', 8), ('beg', 8), ('book', 8), ('changed', 8), ('direction', 8), ('distance', 8), ('dry', 8), ('eagerly', 8), ('everybody', 8), ('exactly', 8), ('fact', 8), ('feel', 8), ('followed', 8), ('gardener', 8), ('happen', 8), ('hard', 8), ('history', 8), ('live', 8), ('lying', 8), ('making', 8), ('meaning', 8), ('mushroom', 8), ('nobody', 8), ('nose', 8), ('noticed', 8), ('opportunity', 8), ('party', 8), ('piece', 8), ('play', 8), ('ready', 8), ('rule', 8), ('running', 8), ('seem', 8), ('shoulder', 8), ('sit', 8), ('slowly', 8), ('sound', 8), ('story', 8), ('twinkle', 8), ('verse', 8), ('watch', 8), ('whiting', 8), ('william', 8), ('window', 8), ('wood', 8), ('yes', 8), ('bat', 7), ('begun', 7), ('bright', 7), ('business', 7), ('cheshire', 7), ('chimney', 7), ('chin', 7), ('deep', 7), ('draw', 7), ('dream', 7), ('drink', 7), ('evidence', 7), ('fall', 7), ('fancy', 7), ('fetch', 7), ('flamingo', 7), ('frightened', 7), ('generally', 7), ('girl', 7), ('golden', 7), ('grin', 7), ('grown', 7), ('hair', 7), ('happened', 7), ('important', 7), ('kind', 7), ('larger', 7), ('learn', 7), ('listen', 7), ('lory', 7), ('manage', 7), ('middle', 7), ('neck', 7), ('nonsense', 7), ('oop', 7), ('open', 7), ('others', 7), ('paw', 7), ('pleased', 7), ('puppy', 7), ('repeat', 7), ('shoe', 7), ('silent', 7), ('sir', 7), ('somebody', 7), ('song', 7), ('soo', 7), ('stood', 7), ('subject', 7), ('surprised', 7), ('tart', 7), ('wondering', 7), ('world', 7), ('almost', 6), ('animal', 6), ('broken', 6), ('cake', 6), ('care', 6), ('chorus', 6), ('croquet', 6), ('cry', 6), ('dish', 6), ('dog', 6), ('dreadfully', 6), ('e', 6), ('egg', 6), ('english', 6), ('exclaimed', 6), ('executed', 6), ('executioner', 6), ('fell', 6), ('forgotten', 6), ('full', 6), ('hour', 6), ('inch', 6), ('jumped', 6), ('late', 6), ('leaf', 6), ('liked', 6), ('loud', 6), ('marked', 6), ('melancholy', 6), ('nice', 6), ('pair', 6), ('pardon', 6), ('pocket', 6), ('politely', 6), ('prize', 6), ('procession', 6), ('roof', 6), ('sentence', 6), ('sharp', 6), ('shriek', 6), ('sing', 6), ('sleep', 6), ('sneezing', 6), ('stand', 6), ('stay', 6), ('stop', 6), ('stupid', 6), ('ten', 6), ('tired', 6), ('told', 6), ('trembling', 6), ('trouble', 6), ('understand', 6), ('upon', 6), ('wow', 6), ('write', 6), ('written', 6), ('youth', 6), ('ah', 5), ('aloud', 5), ('altogether', 5), ('angry', 5), ('arch', 5), ('argument', 5), ('asking', 5), ('become', 5), ('bread-and-butter', 5), ('case', 5), ('confusion', 5), ('corner', 5), ('crowded', 5), ('curiosity', 5), ('cut', 5), ('dare', 5), ('difficulty', 5), ('drew', 5), ('dropped', 5), ('evening', 5), ('execution', 5), ('father', 5), ('fellow', 5), ('finger', 5), ('finish', 5), ('fish', 5), ('friend', 5), ('ground', 5), ('happens', 5), ('height', 5), ('home', 5), ('hookah', 5), ('hot', 5), ('impatiently', 5), ('instantly', 5), ('interesting', 5), ('juror', 5), ('kid', 5), ('knee', 5), ('likely', 5), ('lizard', 5), ('man', 5), ('meant', 5), ('mile', 5), ('morning', 5), ('moved', 5), ('nearer', 5), ('nervous', 5), ('new', 5), ('notice', 5), ('number', 5), ('officer', 5), ('often', 5), ('pack', 5), ('passed', 5), ('person', 5), ('picture', 5), ('plate', 5), ('present', 5), ('quietly', 5), ('remembered', 5), ('reply', 5), ('sadly', 5), ('school', 5), ('seems', 5), ('sha', 5), ('shrill', 5), ('shut', 5), ('sighed', 5), ('simple', 5), ('sleepy', 5), ('sob', 5), ('sometimes', 5), ('sounded', 5), ('speaking', 5), ('stick', 5), ('strange', 5), ('sudden', 5), ('surprise', 5), ('swam', 5), ('swim', 5), ('taking', 5), ('teacup', 5), ('temper', 5), ('treacle', 5), ('twice', 5), ('unimportant', 5), ('usual', 5), ('walk', 5), ('walking', 5), ('water', 5), ('waving', 5), ('whispered', 5), ('wrong', 5), ('young', 5), ('age', 4), ('ala', 4), ('alone', 4), ('ann', 4), ('answered', 4), ('bank', 4), ('beat', 4), ('beau', 4), ('boot', 4), ('bottom', 4), ('bowed', 4), ('box', 4), ('breath', 4), ('busily', 4), ('capital', 4), ('carried', 4), ('catch', 4), ('chance', 4), ('confused', 4), ('consider', 4), ('crowd', 4), ('crumb', 4), ('dead', 4), ('decidedly', 4), ('deeply', 4), ('digging', 4), ('doubt', 4), ('duck', 4), ('earth', 4), ('elbow', 4), ('escape', 4), ('fallen', 4), ('fast', 4), ('fear', 4), ('figure', 4), ('fit', 4), ('fond', 4), ('french', 4), ('frowning', 4), ('grass', 4), ('green', 4), ('growl', 4), ('grunted', 4), ('guinea-pigs', 4), ('hearing', 4), ('held', 4), ('indignantly', 4), ('judge', 4), ('jumping', 4), ('jury-box', 4), ('juryman', 4), ('kitchen', 4), ('lay', 4), ('le', 4), ('led', 4), ('letter', 4), ('mabel', 4), ('managed', 4), ('mary', 4), ('miss', 4), ('natural', 4), ('need', 4), ('noise', 4), ('none', 4), ('ootiful', 4), ('ordered', 4), ('otherwise', 4), ('pale', 4), ('paper', 4), ('particular', 4), ('passage', 4), ('pepper', 4), ('perfectly', 4), ('plan', 4), ('player', 4), ('pointing', 4), ('porpoise', 4), ('puzzling', 4), ('quadrille', 4), ('rabbit-hole', 4), ('reach', 4), ('rose', 4), ('run', 4), ('savage', 4), ('screamed', 4), ('second', 4), ('several', 4), ('severely', 4), ('sharply', 4), ('short', 4), ('shrinking', 4), ('sigh', 4), ('sky', 4), ('sneeze', 4), ('solemnly', 4), ('stuff', 4), ('suppressed', 4), ('taken', 4), ('tale', 4), ('taught', 4), ('telescope', 4), ('thank', 4), ('thimble', 4), ('thoughtfully', 4), ('tiny', 4), ('tongue', 4), ('true', 4), ('twelve', 4), ('twinkling', 4), ('uncomfortable', 4), ('vanished', 4), ('ventured', 4), ('verdict', 4), ('violently', 4), ('wanted', 4), ('week', 4), ('worth', 4), ('writing', 4), ('yer', 4), ("'t", 3), ('advance', 3), ('advantage', 3), ('alive', 3), ('allow', 3), ('anxious', 3), ('attending', 3), ('beast', 3), ('beheaded', 3), ('behind', 3), ('blow', 3), ('boy', 3), ('branch', 3), ('breathe', 3), ('bring', 3), ('bringing', 3), ('brought', 3), ('butter', 3), ('candle', 3), ('card', 3), ('carefully', 3), ('caucus-race', 3), ('caught', 3), ('cause', 3), ('cautiously', 3), ('certain', 3), ('checked', 3), ('cheered', 3), ('choked', 3), ('civil', 3), ('confusing', 3), ('considered', 3), ('considering', 3), ('courage', 3), ('crab', 3), ('crash', 3), ('croquet-ground', 3), ('crossed', 3), ('crown', 3), ('dark', 3), ('decided', 3), ('delight', 3), ('dull', 3), ('eager', 3), ('eaglet', 3), ('easily', 3), ('edge', 3), ('edition', 3), ('effect', 3), ('expecting', 3), ('explanation', 3), ('faster', 3), ('filled', 3), ('finding', 3), ('fire', 3), ('floor', 3), ('fly', 3), ('folded', 3), ('follows', 3), ('forgetting', 3), ('free', 3), ('fun', 3), ('funny', 3), ('fur', 3), ('fury', 3), ('general', 3), ('gently', 3), ('grand', 3), ('grave', 3), ('gravely', 3), ('grinned', 3), ('guess', 3), ('guessed', 3), ('guest', 3), ('handed', 3), ('hanging', 3), ('hedge', 3), ('hoarse', 3), ('holding', 3), ('honour', 3), ('hope', 3), ('hoping', 3), ('howling', 3), ('hungry', 3), ('hunting', 3), ('hurt', 3), ('hush', 3), ("i'd", 3), ("i'll", 3), ("i've", 3), ('immediately', 3), ('impossible', 3), ('instance', 3), ('instead', 3), ('jaw', 3), ('joined', 3), ('kick', 3), ('knocking', 3), ('knowledge', 3), ('lady', 3), ('later', 3), ('leg', 3), ('line', 3), ('list', 3), ('listening', 3), ('lived', 3), ('livery', 3), ('lock', 3), ('longer', 3), ('lost', 3), ('loudly', 3), ('love', 3), ('manner', 3), ('mark', 3), ('master', 3), ('mentioned', 3), ('met', 3), ('mistake', 3), ('move', 3), ('moving', 3), ('muchness', 3), ('music', 3), ('muttering', 3), ('nibbling', 3), ('night', 3), ('notion', 3), ('nurse', 3), ('nursing', 3), ('obliged', 3), ('opening', 3), ('order', 3), ('out-of-the-way', 3), ('owl', 3), ('panther', 3), ('part', 3), ('passion', 3), ('pat', 3), ('pattering', 3), ('peeped', 3), ('picked', 3), ('pity', 3), ('pleaded', 3), ('poison', 3), ('possibly', 3), ('pray', 3), ('pressed', 3), ('proper', 3), ('putting', 3), ('reading', 3), ('real', 3), ('remained', 3), ('remarking', 3), ('repeating', 3), ('rose-tree', 3), ('sad', 3), ('sense', 3), ('settled', 3), ('shaking', 3), ('shaped', 3), ('shore', 3), ('show', 3), ('sighing', 3), ('simply', 3), ('slipped', 3), ('smaller', 3), ('snail', 3), ('sobbing', 3), ('solemn', 3), ('somewhere', 3), ('sorrow', 3), ('spectacle', 3), ('speech', 3), ('spread', 3), ('stair', 3), ('staring', 3), ('stopped', 3), ('succeeded', 3), ('suit', 3), ('sulky', 3), ('taste', 3), ('tasted', 3), ('throw', 3), ('timid', 3), ('to-day', 3), ('toe', 3), ('tortoise', 3), ('tossing', 3), ('trumpet', 3), ('tucked', 3), ('unfortunate', 3), ('upset', 3), ('venture', 3), ('washing', 3), ('watching', 3), ('whisker', 3), ('whisper', 3), ('wonderland', 3), ('worse', 3), ('wrote', 3), ('year', 3), ('yesterday', 3), ('_i_', 2), ('absurd', 2), ('accident', 2), ('account', 2), ('addressed', 2), ('advice', 2), ('advisable', 2), ('afterwards', 2), ('ago', 2), ('agree', 2), ('alarm', 2), ('along', 2), ('already', 2), ('also', 2), ('anger', 2), ('apple', 2), ('archbishop', 2), ('ashamed', 2), ('assembled', 2), ('atom', 2), ('attempt', 2), ('authority', 2), ('bad', 2), ('bark', 2), ('barrowful', 2), ('bear', 2), ('beating', 2), ('beautifully', 2), ('became', 2), ('bed', 2), ('belongs', 2), ('bend', 2), ('besides', 2), ('bite', 2), ('blast', 2), ('blew', 2), ('body', 2), ('bone', 2), ('break', 2), ('brightened', 2), ('broke', 2), ('brown', 2), ('burn', 2), ('busy', 2), ('carrying', 2), ('catching', 2), ('cauldron', 2), ('caused', 2), ('changing', 2), ('choice', 2), ('chose', 2), ('claw', 2), ('clear', 2), ('clever', 2), ('clock', 2), ('closed', 2), ('coaxing', 2), ('collected', 2), ('comfit', 2), ('concert', 2), ('concluded', 2), ('conclusion', 2), ('conqueror', 2), ('constant', 2), ('contemptuously', 2), ('cool', 2), ('couple', 2), ('courtier', 2), ('crimson', 2), ('cross-examine', 2), ('cup', 2), ('cupboard', 2), ('curiouser', 2), ('curled', 2), ('cushion', 2), ('custody', 2), ('dancing', 2), ('declare', 2), ('delighted', 2), ('delightful', 2), ('denied', 2), ('deny', 2), ('difficult', 2), ('dinn', 2), ('dinner', 2), ('dipped', 2), ('directed', 2), ('directly', 2), ('disappeared', 2), ('dispute', 2), ('distant', 2), ('doth', 2), ('doubtful', 2), ('doubtfully', 2), ('drawling', 2), ('dreadful', 2), ('drive', 2), ('drunk', 2), ('earl', 2), ('earnestly', 2), ('easy', 2), ('edwin', 2), ('eel', 2), ('encouraging', 2), ('entangled', 2), ('entirely', 2), ('excellent', 2), ('experiment', 2), ('extra', 2), ('extraordinary', 2), ('extremely', 2), ('falling', 2), ('fancied', 2), ('fashion', 2), ('feather', 2), ('feeble', 2), ('ferret', 2), ('fight', 2), ('fine', 2), ('fish-footman', 2), ('flapper', 2), ('flat', 2), ('flower', 2), ('follow', 2), ('footstep', 2), ('forehead', 2), ('forget', 2), ('forgot', 2), ('fountain', 2), ('fright', 2), ('frog', 2), ('front', 2), ('furrow', 2), ('giddy', 2), ('giving', 2), ('goldfish', 2), ('goose', 2), ('guinea-pig', 2), ('half-past', 2), ('hate', 2), ('heavy', 2), ('hint', 2), ('hit', 2), ('humbly', 2), ('hurriedly', 2), ("ill.'", 2), ('imagine', 2), ('interrupting', 2), ('introduce', 2), ('invitation', 2), ('invited', 2), ("it'll", 2), ('jar', 2), ('keeping', 2), ('kill', 2), ('kindly', 2), ('knife', 2), ('knot', 2), ('knowing', 2), ('label', 2), ('laid', 2), ('lap', 2), ('lasted', 2), ('latitude', 2), ('laughed', 2), ('laughing', 2), ('law', 2), ('leaning', 2), ('learning', 2), ('learnt', 2), ('lie', 2), ('living', 2), ('lonely', 2), ('longed', 2), ('longitude', 2), ('lovely', 2), ('luckily', 2), ('maybe', 2), ('meekly', 2), ('meet', 2), ('mercia', 2), ('merely', 2), ('message', 2), ('miserable', 2), ('missed', 2), ('mixed', 2), ('month', 2), ('morcar', 2), ('mostly', 2), ('mustard', 2), ('muttered', 2), ('mystery', 2), ('narrow', 2), ('neatly', 2), ('nibbled', 2), ('nicely', 2), ('northumbria', 2), ('note-book', 2), ('nowhere', 2), ("o'clock", 2), ('occurred', 2), ('offer', 2), ('older', 2), ('ordering', 2), ('ornamented', 2), ('painting', 2), ('panting', 2), ('parchment', 2), ('paris', 2), ('partner', 2), ('patiently', 2), ('pebble', 2), ('pencil', 2), ('pennyworth', 2), ('persisted', 2), ('personal', 2), ('picking', 2), ('pie', 2), ('pinch', 2), ('pinched', 2), ('playing', 2), ('pleasure', 2), ('plenty', 2), ('position', 2), ('presently', 2), ('prisoner', 2), ('proceed', 2), ('prof', 2), ('proud', 2), ('proved', 2), ('purring', 2), ('quarrelling', 2), ('quick', 2), ('quickly', 2), ('quiet', 2), ('race', 2), ('railway', 2), ('raised', 2), ('rapidly', 2), ('rattling', 2), ('raven', 2), ('raving', 2), ('recovered', 2), ('red', 2), ('regular', 2), ('relief', 2), ('remarkable', 2), ('removed', 2), ('resting', 2), ('returned', 2), ('riddle', 2), ('ridge', 2), ('ring', 2), ('ringlet', 2), ('rise', 2), ('rome', 2), ('row', 2), ('royal', 2), ('rubbing', 2), ('rude', 2), ('rush', 2), ('safe', 2), ('salt', 2), ('sand', 2), ('sang', 2), ('saucepan', 2), ('save', 2), ('scream', 2), ('scroll', 2), ('secondly', 2), ('sell', 2), ('sending', 2), ('sensation', 2), ('sent', 2), ('sh', 2), ('shark', 2), ('shilling', 2), ('shorter', 2), ('shouting', 2), ('shower', 2), ('showing', 2), ('shutting', 2), ('signed', 2), ('singer', 2), ('singing', 2), ('smallest', 2), ('smile', 2), ('smiled', 2), ('smiling', 2), ('smoking', 2), ('snatch', 2), ('sooner', 2), ('sorrowful', 2), ('splashing', 2), ('spoon', 2), ('squeaking', 2), ('stamping', 2), ('started', 2), ('startled', 2), ('stirring', 2), ('stole', 2), ('stoop', 2), ('stretched', 2), ('stretching', 2), ('struck', 2), ('sulkily', 2), ('summer', 2), ('sun', 2), ('swimming', 2), ('taller', 2), ('tea-party', 2), ('tea-time', 2), ('telling', 2), ("that's", 2), ('thistle', 2), ('thoroughly', 2), ('though', 2), ('thousand', 2), ('three-legged', 2), ('threw', 2), ('throat', 2), ('throwing', 2), ('thump', 2), ('tiptoe', 2), ('treacle-well', 2), ('treading', 2), ('trembled', 2), ('triumphantly', 2), ('trotting', 2), ('tumbling', 2), ('tut', 2), ('twenty-four', 2), ('twist', 2), ('uglification', 2), ('ugly', 2), ('undertone', 2), ('uneasily', 2), ('unfolded', 2), ('unhappy', 2), ('unpleasant', 2), ('unrolled', 2), ('useful', 2), ('using', 2), ('usually', 2), ('violent', 2), ('wag', 2), ('wake', 2), ('wandered', 2), ('wandering', 2), ('wash', 2), ('wasting', 2), ('watched', 2), ('weak', 2), ('wet', 2), ('wide', 2), ('wig', 2), ('wild', 2), ('wildly', 2), ('wind', 2), ('wine', 2), ('wink', 2), ('wise', 2), ('woman', 2), ('wonderful', 2), ('wretched', 2), ('yawned', 2), ('yawning', 2), ("'em", 1), ('a-piece', 1), ('abide', 1), ('able', 1), ('absence', 1), ('acceptance', 1), ('accidentally', 1), ('accounting', 1), ('accusation', 1), ('accustomed', 1), ('ache', 1), ('act', 1), ('actually', 1), ('ada', 1), ('adding', 1), ('addressing', 1), ('adjourn', 1), ('adoption', 1), ('advise', 1), ('affair', 1), ('affectionately', 1), ('afford', 1), ('afore', 1), ('after-time', 1), ('agony', 1), ('ahem', 1), ('alarmed', 1), ('all.', 1), ('altered', 1), ('alternately', 1), ('ambition', 1), ('ancient', 1), ('and-butter', 1), ('annoy', 1), ('annoyed', 1), ('antipathy', 1), ('anywhere', 1), ('appealed', 1), ('appear', 1), ('appearance', 1), ('appearing', 1), ('applause', 1), ('argue', 1), ('argued', 1), ('arithmetic', 1), ('arm-chair', 1), ('arm-in-arm', 1), ('arranged', 1), ('arrived', 1), ('arrow', 1), ('arrum', 1), ('askance', 1), ('ate', 1), ('atheling', 1), ('attempted', 1), ('attended', 1), ('attends', 1), ('audibly', 1), ('australia', 1), ('avoid', 1), ('awfully', 1), ('ax', 1), ('back-somersault', 1), ('bag', 1), ('baked', 1), ('balanced', 1), ('ball', 1), ('banquet', 1), ('barking', 1), ('barley-sugar', 1), ('bathing', 1), ('bawled', 1), ('beak', 1), ('beauti', 1), ('beautify', 1), ('becoming', 1), ('bee', 1), ('begged', 1), ('behead', 1), ('beheading', 1), ('believed', 1), ('bell', 1), ('belong', 1), ('beloved', 1), ('belt', 1), ('bent', 1), ('birthday', 1), ('bitter', 1), ('blacking', 1), ('blade', 1), ('blame', 1), ('bleeds', 1), ('blown', 1), ('boldly', 1), ('book-shelves', 1), ('boon', 1), ('bore', 1), ('bother', 1), ('bound', 1), ('bowing', 1), ('boxed', 1), ('brain', 1), ('brandy', 1), ('brass', 1), ('brave', 1), ('bread-', 1), ('bread-knife', 1), ('breeze', 1), ('bright-eyed', 1), ('bristling', 1), ('brother', 1), ('brush', 1), ('brushing', 1), ('burning', 1), ('burnt', 1), ('burst', 1), ('bursting', 1), ('buttercup', 1), ('buttered', 1), ('butterfly', 1), ('button', 1), ('by-the-bye', 1), ('c', 1), ('cackled', 1), ('calling', 1), ('calmly', 1), ('camomile', 1), ("can't", 1), ('canary', 1), ('canterbury', 1), ('canvas', 1), ('capering', 1), ('cardboard', 1), ('carrier', 1), ('carroll', 1), ('carry', 1), ('cart-horse', 1), ('cartwheel', 1), ("caterpillar's", 1), ('cattle', 1), ('ceiling', 1), ('centre', 1), ('chain', 1), ('chanced', 1), ('character', 1), ('charge', 1), ('chatte', 1), ('cheap', 1), ('cheated', 1), ('cheek', 1), ('cheerfully', 1), ('cherry-tart', 1), ('chief', 1), ('child-life', 1), ('childhood', 1), ('choke', 1), ('choking', 1), ('choosing', 1), ('chop', 1), ('christmas', 1), ('chrysalis', 1), ('chuckled', 1), ('circle', 1), ('circumstance', 1), ('clamour', 1), ('clapping', 1), ('clasped', 1), ('classic', 1), ('clean', 1), ('cleared', 1), ('clearer', 1), ('clearly', 1), ('climb', 1), ('clinging', 1), ('closely', 1), ('closer', 1), ('club', 1), ('coast', 1), ('coil', 1), ('cold', 1), ('collar', 1), ('comfort', 1), ('comfortable', 1), ('comfortably', 1), ('common', 1), ('commotion', 1), ('company', 1), ('complained', 1), ('complaining', 1), ('completely', 1), ('condemn', 1), ('conduct', 1), ('conger-eel', 1), ('conquest', 1), ('consented', 1), ('consultation', 1), ('contempt', 1), ('contemptuous', 1), ('content', 1), ('contradicted', 1), ('cost', 1), ('counting', 1), ('country', 1), ('coward', 1), ('crashed', 1), ('crawled', 1), ('crawling', 1), ('crazy', 1), ('creep', 1), ('crept', 1), ('crocodile', 1), ('croqueted', 1), ('croqueting', 1), ('cross', 1), ('crossly', 1), ('crouched', 1), ('cucumber-frame', 1), ('cucumber-frames', 1), ('cunning', 1), ('cur', 1), ('curl', 1), ('curly', 1), ('currant', 1), ('curtain', 1), ('curtsey', 1), ('curtseying', 1), ('curving', 1), ('custard', 1), ('cutting', 1), ('dainty', 1), ('daisy', 1), ('daisy-chain', 1), ('daresay', 1), ('darkness', 1), ('date', 1), ('daughter', 1), ('day-school', 1), ('death', 1), ('declared', 1), ('deepest', 1), ('delay', 1), ('denial', 1), ('denies', 1), ('denying', 1), ('depends', 1), ('derision', 1), ('deserved', 1), ('despair', 1), ('desperate', 1), ('desperately', 1), ('diamond', 1), ('die', 1), ('died', 1), ('dig', 1), ('diligently', 1), ('disagree', 1), ('disappointment', 1), ('disgust', 1), ('dismay', 1), ('disobey', 1), ('distraction', 1), ('dive', 1), ('dodged', 1), ('doorway', 1), ('double', 1), ('doubled-up', 1), ('doubling', 1), ('downward', 1), ('downwards', 1), ('doze', 1), ('dozing', 1), ('draggled', 1), ('drawing', 1), ('drawling-master', 1), ('dreamed', 1), ('dreaming', 1), ('dreamy', 1), ('dressed', 1), ('dried', 1), ('driest', 1), ('drinking', 1), ('dripping', 1), ('drop', 1), ('dropping', 1), ('drowned', 1), ('dunce', 1), ('eating', 1), ('eats', 1), ('edgar', 1), ('education', 1), ('eh', 1), ('either', 1), ('elegant', 1), ('eleventh', 1), ('elsie', 1), ('emphasis', 1), ('empty', 1), ('encourage', 1), ('encouraged', 1), ('ending', 1), ('energetic', 1), ('engaged', 1), ('england', 1), ('engraved', 1), ('enjoy', 1), ('enormous', 1), ('entrance', 1), ('esq', 1), ('est', 1), ('evidently', 1), ('exact', 1), ('examining', 1), ('exclamation', 1), ('execute', 1), ('executes', 1), ('existence', 1), ('expected', 1), ('explained', 1), ('expressing', 1), ('expression', 1), ('eyelid', 1), ('eyes.', 1), ('fading', 1), ('failure', 1), ('faint', 1), ('fainting', 1), ('faintly', 1), ('fair', 1), ('fairly', 1), ('fairy-tales', 1), ('familiarly', 1), ('family', 1), ('fancying', 1), ('fanned', 1), ('fanning', 1), ('farm-yard', 1), ('farmer', 1), ('farther', 1), ('fat', 1), ('favoured', 1), ('favourite', 1), ('feared', 1), ('feebly', 1), ('fender', 1), ('fidgeted', 1), ('field', 1), ('fifteen', 1), ('fifteenth', 1), ('fifth', 1), ('fig', 1), ('fighting', 1), ('fill', 1), ('finishing', 1), ('fire-irons', 1), ('fireplace', 1), ('fitted', 1), ('fix', 1), ('fixed', 1), ('flame', 1), ('flashed', 1), ('flavour', 1), ('flew', 1), ('flinging', 1), ('flock', 1), ('flower-beds', 1), ('flower-pot', 1), ('flown', 1), ('flung', 1), ('flurry', 1), ('flustered', 1), ('fluttered', 1), ('flying', 1), ('folding', 1), ('foolish', 1), ('forepaw', 1), ('fork', 1), ('form', 1), ('fortunately', 1), ('forty-two', 1), ('forward', 1), ('fourteenth', 1), ('fourth', 1), ('france', 1), ('frighten', 1), ('frog-footman', 1), ('frontispiece', 1), ('frying-pan', 1), ('ful', 1), ('fulcrum', 1), ('fumbled', 1), ('furious', 1), ('furiously', 1), ('gained', 1), ('gallon', 1), ('gather', 1), ('gay', 1), ('gazing', 1), ('geography', 1), ('given', 1), ('glanced', 1), ('glaring', 1), ('globe', 1), ('gloomily', 1), ('good-', 1), ('good-bye', 1), ('good-naturedly', 1), ('graceful', 1), ('grammar', 1), ('grant', 1), ('gravy', 1), ('grazed', 1), ('grew', 1), ('grey', 1), ('grief', 1), ('grinning', 1), ('growled', 1), ('growling', 1), ('grumbled', 1), ('grunt', 1), ('guard', 1), ('guilt', 1), ('handsome', 1), ('handwriting', 1), ('happening', 1), ('happy', 1), ('harm', 1), ('haste', 1), ('hat', 1), ('hatching', 1), ('hated', 1), ('heap', 1), ('hearth', 1), ('hearthrug', 1), ('heel', 1), ('helped', 1), ('helpless', 1), ('herald', 1), ('hid', 1), ('hide', 1), ('highest', 1), ('hippopotamus', 1), ('hiss', 1), ('hjckrrh', 1), ('hm', 1), ('hoarsely', 1), ('holiday', 1), ('hollow', 1), ('honest', 1), ('hoped', 1), ('hopeful', 1), ('hopeless', 1), ('hot-tempered', 1), ('housemaid', 1), ('howled', 1), ('humble', 1), ('hundred', 1), ('hung', 1), ('hurrying', 1), ('idiot', 1), ('idiotic', 1), ('ignorant', 1), ('ii', 1), ('iii', 1), ('imitated', 1), ('immediate', 1), ('immense', 1), ('impatient', 1), ('impertinent', 1), ('improve', 1), ('incessantly', 1), ('inclined', 1), ('indignant', 1), ('injure', 1), ('ink', 1), ('inkstand', 1), ('inquired', 1), ('inquisitively', 1), ('inside', 1), ('insolence', 1), ('insult', 1), ('interest', 1), ('interrupt', 1), ('introduced', 1), ('invent', 1), ('invented', 1), ('involved', 1), ('inwards', 1), ('irritated', 1), ('iv', 1), ('ix', 1), ('jack-in-the-box', 1), ('jelly-fish', 1), ('jogged', 1), ('journey', 1), ('joy', 1), ('judging', 1), ('jury-men', 1), ('justice', 1), ('kettle', 1), ('killing', 1), ('kiss', 1), ('kissed', 1), ('kneel', 1), ('knelt', 1), ('knock', 1), ('knocked', 1), ('known', 1), ('knuckle', 1), ('labelled', 1), ('lacie', 1), ('lad', 1), ('ladder', 1), ('lamp', 1), ('land', 1), ('languid', 1), ('largest', 1), ('lark', 1), ('lastly', 1), ('lately', 1), ('latin', 1), ('laugh', 1), ('laughter', 1), ('lazily', 1), ('lazy', 1), ('leader', 1), ('leading', 1), ('leant', 1), ('leap', 1), ('learned', 1), ('leaving', 1), ('ledge', 1), ('lefthand', 1), ('length', 1), ('lessen', 1), ('lesson-book', 1), ('lesson-books', 1), ('lest', 1), ('lewis', 1), ('licking', 1), ('lifted', 1), ('limb', 1), ('linked', 1), ('lip', 1), ('listened', 1), ('listener', 1), ('lit', 1), ("lizard's", 1), ('locked', 1), ('lodging', 1), ('london', 1), ('look-out', 1), ('looking-', 1), ('loose', 1), ('lose', 1), ('losing', 1), ('louder', 1), ('loveliest', 1), ('loving', 1), ('low-spirited', 1), ('lower', 1), ('lowing', 1), ('lullaby', 1), ("ma'am", 1), ('machine', 1), ('magic', 1), ('magpie', 1), ('mallet', 1), ('managing', 1), ('map', 1), ('marched', 1), ('marmalade', 1), ("me'", 1), ('meal', 1), ('meanwhile', 1), ('measure', 1), ('meat', 1), ('meeting', 1), ('memorandum', 1), ('memory', 1), ('merrily', 1), ('milk', 1), ('milk-jug', 1), ('millennium', 1), ('minded', 1), ('minding', 1), ('mineral', 1), ('mischief', 1), ('moderate', 1), ('modern', 1), ('moon', 1), ('morsel', 1), ('mournful', 1), ('mournfully', 1), ('mouse-traps', 1), ('muddle', 1), ('multiplication', 1), ('murder', 1), ('murdering', 1), ('muscular', 1), ('mustard-mine', 1), ('nasty', 1), ('natured', 1), ('nay', 1), ('near', 1), ('neat', 1), ('neighbour', 1), ('neighbouring', 1), ('nest', 1), ('never-ending', 1), ('nevertheless', 1), ('newspaper', 1), ('night-air', 1), ('nile', 1), ('nodded', 1), ('norman', 1), ('noticing', 1), ('obstacle', 1), ('occasional', 1), ('occasionally', 1), ('odd', 1), ('offend', 1), ('ointment', 1), ('oldest', 1), ('onion', 1), ('opinion', 1), ('opposite', 1), ('orange', 1), ('ou', 1), ('outside', 1), ('overhead', 1), ('oyster', 1), ('pace', 1), ('paint', 1), ('panted', 1), ('pardoned', 1), ('pas', 1), ('passing', 1), ('passionate', 1), ('past', 1), ('patience', 1), ('patriotic', 1), ('patted', 1), ('pattern', 1), ('pause', 1), ('paused', 1), ('peeping', 1), ('peering', 1), ('peg', 1), ('penny', 1), ('pepper-box', 1), ('permitted', 1), ('pictured', 1), ('pie-crust', 1), ('pig-baby', 1), ('pine-apple', 1), ('pink', 1), ('piteous', 1), ('pitied', 1), ('placed', 1), ('plainly', 1), ('planning', 1), ('played', 1), ('plea', 1), ('pleasant', 1), ('pleasanter', 1), ('pleasing', 1), ('pointed', 1), ('poker', 1), ('poky', 1), ('pop', 1), ('pope', 1), ('positively', 1), ('possible', 1), ('pound', 1), ('pour', 1), ('poured', 1), ('powdered', 1), ('practice', 1), ('precious', 1), ('presented', 1), ('pressing', 1), ('pretend', 1), ('pretending', 1), ('pretext', 1), ('prettier', 1), ('pretty', 1), ('prevent', 1), ('printed', 1), ('prison', 1), ('produced', 1), ('producing', 1), ('promise', 1), ('promised', 1), ('promising', 1), ('pronounced', 1), ('proposal', 1), ('prosecute', 1), ('protection', 1), ('prove', 1), ('provoking', 1), ('puffed', 1), ('pulled', 1), ('pulling', 1), ('pun', 1), ('punching', 1), ('punished', 1), ('purple', 1), ('purpose', 1), ('pus', 1), ('push', 1), ('puzzle', 1), ('quarrel', 1), ('quarrelled', 1), ('queer-', 1), ('queer-looking', 1), ('queerest', 1), ('questions.', 1), ('quicker', 1), ('quiver', 1), ("rabbit'", 1), ('race-course', 1), ('raising', 1), ('rapped', 1), ('rat', 1), ('rat-hole', 1), ('rattle', 1), ('raw', 1), ('reaching', 1), ('readily', 1), ('reality', 1), ('rearing', 1), ('reasonable', 1), ('received', 1), ('recognised', 1), ('red-hot', 1), ('reduced', 1), ('reed', 1), ('reeling', 1), ('refreshment', 1), ('refused', 1), ('relieved', 1), ('remain', 1), ('remaining', 1), ('remedy', 1), ('remembering', 1), ('reminding', 1), ('resource', 1), ('respect', 1), ('respectable', 1), ('respectful', 1), ('result', 1), ('retire', 1), ('returning', 1), ('rich', 1), ('riddles.', 1), ('ridiculous', 1), ('right-hand', 1), ('righthand', 1), ('rightly', 1), ('riper', 1), ('rippling', 1), ('rising', 1), ('roared', 1), ('roast', 1), ('rock', 1), ('root', 1), ('rope', 1), ('rosetree', 1), ('roughly', 1), ('rubbed', 1), ('rudeness', 1), ('rumbling', 1), ('rushed', 1), ('rustled', 1), ('rustling', 1), ('sage', 1), ('salmon', 1), ('saucer', 1), ('scale', 1), ('scaly', 1), ('schoolroom', 1), ('scolded', 1), ('scrambling', 1), ('scratching', 1), ('screaming', 1), ('sea-shore', 1), ('seal', 1), ('seaography', 1), ('search', 1), ('seaside', 1), ('seated', 1), ('secret', 1), ('seeing', 1), ('seldom', 1), ('send', 1), ('sends', 1), ('sentenced', 1), ('series', 1), ('seriously', 1), ('setting', 1), ('settle', 1), ('settling', 1), ('severity', 1), ('shade', 1), ('shake', 1), ('shakespeare', 1), ('shape', 1), ('share', 1), ('shared', 1), ('sharing', 1), ('shedding', 1), ('sheep-', 1), ('shelf', 1), ('shepherd', 1), ('shifting', 1), ('shingle', 1), ('shining', 1), ('shiny', 1), ('shiver', 1), ('shock', 1), ('shrieked', 1), ('shrimp', 1), ('shrink', 1), ('shy', 1), ('shyly', 1), ('sign', 1), ('signifies', 1), ('signify', 1), ('simpleton', 1), ('sink', 1), ('sits', 1), ('sixpence', 1), ('sixteenth', 1), ('skimming', 1), ('skirt', 1), ('skurried', 1), ('sky-rocket', 1), ('slate-pencil', 1), ('slightest', 1), ('slippery', 1), ('sluggard', 1), ('smoke', 1), ('snappishly', 1), ('sneezed', 1), ('snorting', 1), ('snout', 1), ('sobbed', 1), ('soft', 1), ('softly', 1), ('sol', 1), ('solid', 1), ('somehow', 1), ('someone', 1), ('somersault', 1), ('son', 1), ('soothing', 1), ('sorry', 1), ('sour', 1), ('spade', 1), ('speaker', 1), ('speed', 1), ('spell', 1), ('spite', 1), ('splash', 1), ('splashed', 1), ('splendidly', 1), ('spoken', 1), ('spot', 1), ('sprawling', 1), ('spreading', 1), ('squeaked', 1), ('squeeze', 1), ('squeezed', 1), ('stalk', 1), ('standing', 1), ('star-fish', 1), ('state', 1), ('station', 1), ('steady', 1), ('steam-engine', 1), ('sternly', 1), ('stiff', 1), ('stigand', 1), ('stingy', 1), ('stocking', 1), ('stolen', 1), ('stool', 1), ('stopping', 1), ('straight', 1), ('straightened', 1), ('straightening', 1), ('strength', 1), ('string', 1), ('stupidest', 1), ('stupidly', 1), ('subdued', 1), ('submitted', 1), ('suet', 1), ('sugar', 1), ('supple', 1), ('suppress', 1), ('swallow', 1), ('swallowed', 1), ('swallowing', 1), ('sweet-tempered', 1), ('tea-things', 1), ('tea-tray', 1), ('teaching', 1), ('teapot', 1), ('tease', 1), ('teeth', 1), ('term', 1), ('terribly', 1), ('terrier', 1), ('terror', 1), ('thanked', 1), ('thatched', 1), ("there's", 1), ('therefore', 1), ("they'll", 1), ("they're", 1), ('thick', 1), ('thin', 1), ('thirteen', 1), ('throne', 1), ('thrown', 1), ('thunder', 1), ('thunderstorm', 1), ('tidy', 1), ('tie', 1), ('tied', 1), ('tight', 1), ('tillie', 1), ('tinkling', 1), ('tipped', 1), ('tittered', 1), ('to-night', 1), ('toast', 1), ('today', 1), ('toffee', 1), ('tomorrow', 1), ('toss', 1), ('touch', 1), ('tougher', 1), ('towards', 1), ('toy', 1), ('trampled', 1), ('treat', 1), ('treated', 1), ('tremble', 1), ('tremulous', 1), ('trick', 1), ('trickling', 1), ('trim', 1), ('trot', 1), ('trust', 1), ('truth', 1), ('truthful', 1), ('tulip-roots', 1), ('tumbled', 1), ('tunnel', 1), ('tureen', 1), ('turkey', 1), ('turn-up', 1), ("turtle's", 1), ('twelfth', 1), ('twentieth', 1), ('twenty', 1), ('twinkled', 1), ('ugh', 1), ('uglify', 1), ('uglifying', 1), ('unable', 1), ('uncivil', 1), ('uncomfortably', 1), ('uncommon', 1), ('uncommonly', 1), ('uncorked', 1), ('underneath', 1), ('understood', 1), ('undo', 1), ('undoing', 1), ('uneasy', 1), ('unjust', 1), ('unlocking', 1), ('untwist', 1), ('unusually', 1), ('unwillingly', 1), ('upright', 1), ('upsetting', 1), ('upstairs', 1), ('usurpation', 1), ('v', 1), ('vague', 1), ('vanishing', 1), ('variation', 1), ('various', 1), ('vegetable', 1), ('velvet', 1), ('vi', 1), ('vii', 1), ('viii', 1), ('vinegar', 1), ('violence', 1), ('visit', 1), ('vote', 1), ('vulgar', 1), ('w.', 1), ('waist', 1), ('waistcoat-', 1), ('waistcoat-pocket', 1), ('wait', 1), ('walrus', 1), ('wander', 1), ('warning', 1), ('waste', 1), ('water-well', 1), ('wearily', 1), ('welcome', 1), ('wept', 1), ('whereupon', 1), ('whistle', 1), ('whistling', 1), ('wider', 1), ('wife', 1), ('win', 1), ('wing', 1), ('winter', 1), ('wit', 1), ('woke', 1), ('wondered', 1), ('wooden', 1), ('wore', 1), ('worm', 1), ('worried', 1), ('worry', 1), ('wrapping', 1), ('wriggling', 1), ('writhing', 1), ('writing-desk', 1), ('writing-desks', 1), ('x', 1), ('xi', 1), ('xii', 1), ('yard', 1), ('ye', 1), ('yelled', 1), ('yelp', 1), ('zealand', 1), ('zigzag', 1)]
    The 5 most frequent tokens are: ('said', 462), ('alice', 399), ("n't", 208), ('little', 128), ('know', 89)
    The least frequent tokens are just too many...
    
    In this progress I use lemmatization since I think it contains a little bit more imformation than stemming
"""