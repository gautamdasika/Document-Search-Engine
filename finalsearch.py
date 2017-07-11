import nltk,math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from math import log10,sqrt
from collections import Counter
stemmer = PorterStemmer()
tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
import os
corpusroot = './presidential_debates'#My subdiretory name
vectors={}                          #tf-idf vectors for all documents
df=Counter()                        #storage for document frequency
tfs={}                              #permanent storage for tfs of all tokens in all documents
lengths=Counter()                   #used for calculating lengths of documents
postings_list={}                    #posting list storage for each token in the corpus
st_tokens=[]
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()                                                #given code for reading files and converting the case
    tokens = tokenizer.tokenize(doc)                                 #tokenizing each document
    sw=stopwords.words('english')
    tokens = [stemmer.stem(token) for token in tokens if token not in sw]               #removing stopwords and performing stemming
    tf=Counter(tokens)
    df+=Counter(list(set(tokens)))
    tfs[filename]=tf.copy()                                          #making a copy of tf into tfs for that filename
    tf.clear()                                                       #clearing tf so that the next document will have an empty tf

def calWeight(filename, token):                                      #returns the weight of a token in a document without normalizing
    idf=getidf(token)
    return (1+log10(tfs[filename][token]))*idf                       #tfs has the logs of term frequencies of all docs in a multi-level dict

def getidf(token):
    if df[token]==0:
        return -1
    return log10(len(tfs)/df[token])                #len(tfs) returns no. of docs; df[token] returns the token's document frequency

#loop for calculating tf-idf vectors and lengths of documents
for filename in tfs:
    vectors[filename]=Counter()                     #initializing the tf-idf vector for each doc
    length=0
    for token in tfs[filename]:
        weight = calWeight(filename, token)         #calWeight calculates the weight of a token in a doc without normalization
        vectors[filename][token]=weight             #this is the weight of a token in a file
        length += weight**2                         #calculating length for normalizing later
    lengths[filename]=math.sqrt(length)

#loop for normalizing the weights
for filename in vectors:
    for token in vectors[filename]:
        vectors[filename][token]= vectors[filename][token] / lengths[filename]      #dividing weights by the document's length
        if token not in postings_list:
            postings_list[token]=Counter()
        postings_list[token][filename]=vectors[filename][token]                     #copying the normalized value into the posting list
def getweight(filename,token):
    return vectors[filename][token]             #returns normalized weight of a token in a document

def query(qstring):                             #function that returns the best match for a query
    qstring=qstring.lower()                     #converting the words to lower case
    qtf={}
    qlength=0
    flag=0
    loc_docs={}
    tenth={}
    cos_sims=Counter()                          #initializing a counter for calculating cosine similarity b/w a token and a doc
    for token in qstring.split():
        token=stemmer.stem(token)               #stemming the token using PorterStemmer
        if token not in postings_list:          #if the token doesn't exist in vocabulary,ignore it (this includes stopwords removal)
            continue
        if getidf(token)==0:                    #if a token has idf = 0, all values in its postings list are zero. max 10 will be chosen randomly
            loc_docs[token], weights = zip(*postings_list[token].most_common())         #to avoid that, we store all docs
        else:
            loc_docs[token],weights = zip(*postings_list[token].most_common(10))        #taking top 10 in postings list
        tenth[token]=weights[9]                                                         #storing the upper bound of each token
        if flag==1:
            commondocs=set(loc_docs[token]) & commondocs                                #commondocs keeps track of docs that have all tokens
        else:
            commondocs=set(loc_docs[token])
            flag=1
        qtf[token]=1+log10(qstring.count(token))    #updating term freq of token in query
        qlength+=qtf[token]**2                      #calculating length for normalizing the query tf later
    qlength=sqrt(qlength)
    for doc in vectors:
        cos_sim=0
        for token in qtf:
            if doc in loc_docs[token]:
                cos_sim = cos_sim + (qtf[token] / qlength) * postings_list[token][doc]       #calculate actual score if document is in top 10
            else:
                cos_sim = cos_sim + (qtf[token] / qlength) * tenth[token]                    #otherwise, calculate its upper bound score
        cos_sims[doc]=cos_sim
    max=cos_sims.most_common(1)                                                              #seeing which doc has the max value
    ans,wght=zip(*max)
    try:
        if ans[0] in commondocs:                                                             #if doc has actual score, return score
            return ans[0],wght[0]
        else:
            return "fetch more",0                                                            #if upperbound score is greater, return fetch more
    except UnboundLocalError:                                                                #if none of the tokens are in vocabulary, return none
        return "None",0


"""
References:
https://docs.python.org/3/library/
www.stackoverflow.com
www.nltk.org
https://docs.python.org/3/library/collections.html#collections.Counter
https://www.tutorialspoint.com/python/python_dictionary.htm
www.python.org"""

print("(%s, %.12f)" % query("health insurance wall street"))
print("(%s, %.12f)" % query("security conference ambassador"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
print("(%s, %.12f)" % query("terror attack"))
print("(%s, %.12f)" % query("vector entropy"))
