# Document-Search-Engine
Used Python, NLTK, NLP techniques to make a search engine that ranks documents based on search keyword, based on TF-IDF weights and cosine similarity
CSE 4334/5334 Programming Assignment 1 (P1)
Fall 2016
Due: 11:59pm Central Time, Tuesday, October 18, 2016
In this assignment, you will implement a toy "search engine" in Python. You code will read a corpus and produce TF-IDF vectors for documents in the corpus. Then, given a query string, you code will return the query answer--the document with the highest cosine similarity score for the query. Instead of computing cosine similarity score for each and every document, you will implement a smarter threshold-bounding algorithm which shares the same basic principle as real search engines.
The instructions on this assignment are written in an .ipynb file. You can use the following commands to install the Jupyter notebook viewer. You can use the following commands to install the Jupyter notebook viewer. "pip" is a command for installing Python packages. You are required to use Python 3.5.1 in this project.
pip install jupyter

pip install notebook (You might have to use "sudo" if you are installing them at system level)
To run the Jupyter notebook viewer, use the following command:
jupyter notebook P1.ipynb
The above command will start a webservice at http://localhost:8888/ and display the instructions in the '.ipynb' file.
The same instructions are also available at https://nbviewer.jupyter.org/url/crystal.uta.edu/~cli/cse5334/ipythonnotebook/P1.ipynb
Requirements
This assignment must be done individually. You must implement the whole assignment by yourself. Academic dishonety will have serious consequences.
You can discuss topics related to the assignment with your fellow students. But you are not allowed to discuss/share your solution and code.
Dataset
We use a corpus of all the general election presidential debates from 1960 to 2012. We processed the corpus and provided you a .zip file, which includes 30 .txt files. Each of the 30 files contains the transcript of a debate and is named by the date of the debate. The .zip file can be downloaded from Blackboard ("Course Materials" > "Programming Assignment 1" > "Attached Files: presidential_debates.zip").
Programming Language
You are required to use Python 3.5.1. You are required to submit a single .py file of your code. We will test your code under the particular version of Python 3.5.1. So make sure you develop your code using the same version.
You are expected to use several modules in NLTK--a natural language processing toolkit for Python. NLTK doesn't come with Python by default. You need to install it and "import" it in your .py file. NLTK's website (http://www.nltk.org/index.html) provides a lot of useful information, including a book http://www.nltk.org/book/, as well as installation instructions (http://www.nltk.org/install.html).
In programming assignment 1, other than NLTK, you are not allowed to use any other non-standard Python package. However, you are free to use anything from the the Python Standard Library that comes with Python 3.5.1 (https://docs.python.org/3/library/).
Tasks
You code should accomplish the following tasks:
(1) Read the 30 .txt files, each of which has the transcript of a presidential debate. The following code does it. Make sure to replace "corpusroot" by your directory where the files are stored. In the example below, "corpusroot" is a sub-folder named "presidential_debates" in the folder containing the python file of the code.
In this assignment we ignore the difference between lower and upper cases. So convert the text to lower case before you do anything else with the text. For a query, also convert it to lower case before you answer the query.
In [2]:

import os
corpusroot = './presidential_debates'
for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close() 
    doc = doc.lower()
(2) Tokenize the content of each file. For this, you need a tokenizer. For example, the following piece of code uses a regular expression tokenizer to return all course numbers in a string. Play with it and edit it. You can change the regular expression and the string to observe different output results.
For tokenizing the Presidential debate speeches, let's all use RegexpTokenizer(r'[a-zA-Z]+'). What tokens will it produce? What limitations does it have?
In [6]:

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'[A-Z]{2,3}[1-9][0-9]{3,3}')
tokens = tokenizer.tokenize("CSE4334 and CSE5334 are taught together. IE3013 is an undergraduate course.")
print(tokens)
['CSE4334', 'CSE5334', 'IE3013']
(3) Perform stopword removal on the obtained tokens. NLTK already comes with a stopword list, as a corpus in the "NLTK Data" (http://www.nltk.org/nltk_data/). You need to install this corpus. Follow the instructions at http://www.nltk.org/data.html. You can also find the instruction in this book: http://www.nltk.org/book/ch01.html (Section 1.2 Getting Started with NLTK). Basically, use the following statements in Python interpreter. A pop-up window will appear. Click "Corpora" and choose "stopwords" from the list.
In [3]:

import nltk
nltk.download()
showing info http://www.nltk.org/nltk_data/
Out[3]:
True
After the stopword list is downloaded, you will find a file "english" in folder nltk_data/corpora/stopwords, where folder nltk_data is the download directory in the step above. The file contains 127 stopwords. nltk.corpus.stopwords will give you this list of stopwords. Try the following piece of code.
In [2]:

from nltk.corpus import stopwords
print(stopwords.words('english'))
print(sorted(stopwords.words('english')))
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']
['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 'and', 'any', 'are', 'aren', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', 'd', 'did', 'didn', 'do', 'does', 'doesn', 'doing', 'don', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 'hadn', 'has', 'hasn', 'have', 'haven', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', 'more', 'most', 'mustn', 'my', 'myself', 'needn', 'no', 'nor', 'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', 'she', 'should', 'shouldn', 'so', 'some', 'such', 't', 'than', 'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', 'we', 'were', 'weren', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'won', 'wouldn', 'y', 'you', 'your', 'yours', 'yourself', 'yourselves']
(4) Also perform stemming on the obtained tokens. NLTK comes with a Porter stemmer. Try the following code and learn how to use the stemmer.
In [2]:

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
print(stemmer.stem('studying'))
print(stemmer.stem('vector'))
print(stemmer.stem('entropy'))
print(stemmer.stem('hispanic'))
print(stemmer.stem('ambassador'))
studi
vector
entropi
hispan
ambassador
(5) Using the tokens, compute the TF-IDF vector for each document. Use the following equation that we learned in the lectures to calculate the term weights, in which tt is a token and dd is a document:
wt,d=(1+log10tft,d)×(log10Ndft).
wt,d=(1+log10tft,d)×(log10Ndft).
Note that the TF-IDF vectors should be normalized (i.e., their lengths should be 1).
Represent a TF-IDF vector by a dictionary. The following is a sample TF-IDF vector.
In [37]:

{'sanction': 0.014972337775895645, 'lack': 0.008576372825970286, 'regret': 0.009491784747267843, 'winter': 0.030424375278541155}
Out[37]:
{'lack': 0.008576372825970286,
 'regret': 0.009491784747267843,
 'sanction': 0.014972337775895645,
 'winter': 0.030424375278541155}
(6) Given a query string, calculate the query vector. (Remember to convert it to lower case.) In calculating the query vector, don't consider IDF. I.e., use the following equation to calculate the term weights in the query vector, in which tt is a token and qq is the query:
wt,q=(1+log10tft,q).
wt,q=(1+log10tft,q).
The vector should also be normalized.
(7) Find the document that attains the highest cosine similarity score. If we compute the cosine similarity between the query vector and every document vector, it is too inefficient. Instead, implement the following method:
(7.1) For each token tt that exists in the corpus, construct its postings list---a sorted list in which each element is in the form of (document dd, TF-IDF weight ww). Such an element provides tt's weight ww in document dd. The elements in the list are sorted by weights in descending order.
(7.2) For each token tt in the query, return the top-10 elements in its corresponding postings list. If the token tt doesn't exist in the corpus, ignore it.
(7.3) If a document dd appears in the top-10 elements of every query token, calculate dd's cosine similarity score. Recall that the score is defined as follows. Since dd appears in top-10 of all query tokens, we have all the information to calculate its actual score sim(q,d)sim(q,d).
sim(q,d)=q⃗ ⋅d⃗ =∑t in both q and dwt,q×wt,d.
sim(q,d)=q→⋅d→=∑t in both q and dwt,q×wt,d.
(7.4) If a document dd doesn't appear in the top-10 elements of some query token tt, use the weight in the 10th element as the upper-bound on tt's weight in dd's vector. Hence, we can calculate the upper-bound score for dd using the query tokens' actual and upper-bound weights with respect to dd's vector, as follows.
sim(q,d)⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯=∑t∈T1wt,q×wt,d+∑t∈T2wt,q×wt,d⎯⎯⎯⎯⎯⎯⎯⎯⎯.
sim(q,d)¯=∑t∈T1wt,q×wt,d+∑t∈T2wt,q×wt,d¯.
In the above equation, T1T1 includes query tokens whose top-10 elements contain dd. T2T2 includes query tokens whose top-10 elements do not contain dd. wt,d⎯⎯⎯⎯⎯⎯⎯⎯⎯wt,d¯ is the weight in the 10-th element of tt's postings list. As a special case, for a document dd that doesn't appear in the top-10 elements of any query token tt, its upper-bound score is thus:
sim(q,d)⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯=∑t∈qwt,q×wt,d⎯⎯⎯⎯⎯⎯⎯⎯⎯.
sim(q,d)¯=∑t∈qwt,q×wt,d¯.
(7.5) If a document's actual score is better than or equal to the actual scores and upper-bound scores of all other documents, it is returned as the query answer.
If there isn't such a document, it means we need to go deeper than 10 elements into the postings list of each query token.
What to Submit
Submit through Blackboard your source code in a single .py file. You can use any standard Python library. The only non-standard library/package allowed for this assignment is NLTK. You .py file must define at least the following functions:
query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect to "qstring" according to (7.5). If no document contains any token in the query, return ("None",0). If we need more than 10 elements from each posting list, return ("fetch more",0).
getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. The parameter 'token' is already stemmed. Note the differences between getidf("hispan") and getidf("hispanic") in the examples below.
getweight(filemae,token): return the TF-IDF weight of a token in the document named 'filename'. If the token doesn't exist in the document, return 0. The parameter 'token' is already stemmed. Note that both getweight("1960-10-21.txt","reason") and getweight("2012-10-16.txt","hispanic") return 0, but for different reasons.
Some sample results that we should expect from a correct implementation:
print("(%s, %.12f)" % query("health insurance wall street"))
(2012-10-03.txt, 0.033877975254)
print("(%s, %.12f)" % query("security conference ambassador"))
(1960-10-21.txt, 0.043935804608)
print("(%s, %.12f)" % query("particular constitutional amendment"))
(fetch more, 0.000000000000)
print("(%s, %.12f)" % query("terror attack"))
(2004-09-30.txt, 0.026893338131)
print("(%s, %.12f)" % query("vector entropy"))
(None, 0.000000000000)
print("%.12f" % getweight("2012-10-03.txt","health"))
0.008528366190
print("%.12f" % getweight("1960-10-21.txt","reason"))
0.000000000000
print("%.12f" % getweight("1976-10-22.txt","agenda"))
0.012683891289
print("%.12f" % getweight("2012-10-16.txt","hispan"))
0.023489163449
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
0.000000000000
print("%.12f" % getidf("health"))
0.079181246048
print("%.12f" % getidf("agenda"))
0.363177902413
print("%.12f" % getidf("vector"))
-1.000000000000
print("%.12f" % getidf("reason"))
0.000000000000
print("%.12f" % getidf("hispan"))
0.632023214705
print("%.12f" % getidf("hispanic"))
-1.000000000000
