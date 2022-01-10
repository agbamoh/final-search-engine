from flask import Flask, request, jsonify
import csv

from io import TextIOWrapper
from zipfile import ZipFile
import math
import builtins

import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from time import time
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from google.cloud import storage
from pathlib import Path
from collections import Counter
import pickle
from google.cloud import storage
import nltk
nltk.download('stopwords')

import hashlib
def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()




bucket_name = 'final_project_bucket207978669'
client = storage.Client()
blobs = client.list_blobs(bucket_name)


from inverted_index_gcp import InvertedIndex
from inverted_index_gcp import *

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']


all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

stemmer = PorterStemmer()


NUM_BUCKETS = 124
def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS


def word_count(text, id):
  ''' Count the frequency of each word in `text` (tf) that is not included in
  `all_stopwords` and return entries that will go into our posting lists.
  Parameters:
  -----------
    text: str
      Text of one document
    id: int
      Document id
  Returns:
  --------
    List of tuples
      A list of (token, (doc_id, tf)) pairs
      for example: [("Anarchism", (12, 5)), ...]
  '''
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  countdict={}
  for token in tokens:
    if not token in all_stopwords:
      if token in countdict:
        countdict[token] +=1
      else:
        countdict[token]=1
  tflist=[]
  for key in countdict:
    innerpair=(id,countdict[key])
    pair=(key,innerpair)
    tflist.append(pair)
  return tflist


# In[30]:


def reduce_word_counts(unsorted_pl):
  ''' Returns a sorted posting list by wiki_id.
  Parameters:
  -----------
    unsorted_pl: list of tuples
      A list of (wiki_id, tf) tuples
  Returns:
  --------
    list of tuples
      A sorted posting list.
  '''
  # YOUR CODE HERE

  lst=list(unsorted_pl)
  lst.sort(key=lambda a: a[0])
  return lst



# In[31]:


def calculate_df(postings):
  ''' Takes a posting list RDD and calculate the df for each token.
  Parameters:
  -----------
    postings: RDD
      An RDD where each element is a (token, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each element is a (token, df) pair.
  '''

  newlst=postings.map(lambda x:(x[0],len(x[1])))

  return newlst


# In[32]:


def partition_postings_and_write(postings):
  ''' A function that partitions the posting lists into buckets, writes out
  all posting lists in a bucket to disk, and returns the posting locations for
  each bucket. Partitioning should be done through the use of `token2bucket`
  above. Writing to disk should use the function  `write_a_posting_list`, a
  static method implemented in inverted_index_colab.py under the InvertedIndex
  class.
  Parameters:
  -----------
    postings: RDD
      An RDD where each item is a (w, posting_list) pair.
  Returns:
  --------
    RDD
      An RDD where each item is a posting locations dictionary for a bucket. The
      posting locations maintain a list for each word of file locations and
      offsets its posting list was written to. See `write_a_posting_list` for
      more details.
  '''
  # YOUR CODE HERE
  rdd=postings
  rdd=rdd.map(lambda x: (token2bucket_id(x[0])  ,[(x[0],x[1])]))
  rdd=rdd.reduceByKey(lambda x,y:x+y)
  rdd=rdd.map(lambda x: InvertedIndex.write_a_posting_list(x,bucket_name))
  return rdd



# In[33]:


def calculate_term_freq(term,posing_list):
    sum=0
    for pair in posing_list:
        sum+=pair[1]
    return (term,sum)


# In[34]:





def getDocLength(text,id):
    helpList = []
    tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
    for term in tokens:
      if term in listOfQueryTermsDic.keys():
        helpList.append(term)
    return (id,len(helpList))





# In[ ]:

# full_path = "gs://wikidata_preprocessed/*"
# parquetFile = spark.read.parquet(full_path)
# doc_text_pairs = parquetFile.select("text", "id").rdd


index_body=InvertedIndex.read_index(f'gs://{bucket_name}','relevant_index2')
index_title=InvertedIndex.read_index(f'gs://{bucket_name}','full_index')
index_anchor=InvertedIndex.read_index(f'gs://{bucket_name}','relevant_index')

super_posting_locs = defaultdict(list)
for blob in client.list_blobs(bucket_name, prefix='postings_gcp'):
  if not blob.name.endswith("pickle"):
    continue
  with blob.open("rb") as f:
    posting_locs = pickle.load(f)
    for k, v in posting_locs.items():
      super_posting_locs[k].extend(v)
index_body.posting_locs = super_posting_locs
index_title.posting_locs = super_posting_locs

for key, value in index_body.df.items():
    if value ==0:
        index_body.df[key]=1
        
for key, value in index_body.Length_of_docs.items():
    if value ==0:
        index_body.Length_of_docs[key]=1
def generate_query_tfidf_vector(query_to_search,index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """

    epsilon = .0000001
    total_vocab_size = len(index.term_total)
    Q = np.zeros((total_vocab_size))
    term_vector = list(index.term_total.keys())
    counter = Counter(query_to_search)

    for token in np.unique(query_to_search):

        if token in index.term_total.keys(): #avoid terms that do not appear in the index.
            tf = counter[token]/len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.Length_of_docs))/(df+epsilon),10) #smoothing

            try:
                ind = term_vector.index(token)

                Q[ind] = tf*idf

            except:
                pass

    return Q


# In[ ]:


def get_candidate_documents_and_scores(query_to_search,index,words,pls):
    """
    Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
    and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
    Then it will populate the dictionary 'candidates.'
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the document.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    #print(index.Length_of_docs)
    N = len(index.Length_of_docs)
    if N==0:
        N=1
    #print(N)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]
#             if index.df[term] ==0:
#                 index.df[term]=1


            try:
                normlized_tfidf = [(doc_id,(freq/index.Length_of_docs[doc_id])*math.log(N/index.df[term],10)) for doc_id, freq in list_of_doc]

            except:
                normlized_tfidf=[]


            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id,term)] = candidates.get((doc_id,term), 0) + tfidf

    return candidates


# In[ ]:


def generate_document_tfidf_matrix(query_to_search,index,words,pls):
    """
    Generate a DataFrame `D` of tfidf scores for a given query.
    Rows will be the documents candidates for a given query
    Columns will be the unique terms in the index.
    The value for a given document and term will be its tfidf score.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(index.term_total)
    candidates_scores = get_candidate_documents_and_scores(query_to_search,index,words,pls) #We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = index.term_total.keys()

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D



# In[ ]:


from numpy import dot
from numpy.linalg import norm
from scipy import spatial
def cosine_similarity(D,Q):
    """
    Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
    Generate a dictionary of cosine similarity scores
    key: doc_id
    value: cosine similarity score

    Parameters:
    -----------
    D: DataFrame of tfidf scores.

    Q: vectorized query with tfidf scores

    Returns:
    -----------
    dictionary of cosine similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: cosine similarty score.
    """
    # YOUR CODE HERE

    out={}


    for index, row in D.iterrows():

        a=row.tolist()

        cos_sim = 1 - spatial.distance.cosine(a, Q)

        out[index]=cos_sim

    return out
    #raise NotImplementedError()



# In[ ]:


def get_top_n(sim_dict,N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    #print(sim_dict)
    return sorted([(doc_id, builtins.round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[:N]



# In[ ]:




def get_topN_score_for_queries(queries_to_search,index,N=3):

    """
    Generate a dictionary that gathers for every query its topN score.

    Parameters:
    -----------
    queries_to_search: a dictionary of queries as follows:
                                                        key: query_id
                                                        value: list of tokens.
    index:           inverted index loaded from the corresponding files.
    N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

    Returns:
    -----------
    return: a dictionary of queries and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id, score).
    """
    # YOUR CODE HERE

    dic={}

    
    query=queries_to_search.split()
    words, pls= zip(*index_body.posting_lists_iter(query))
    Q=generate_query_tfidf_vector(query,index)
    D=generate_document_tfidf_matrix(query,index,words,pls)
    consine=cosine_similarity(D,Q)
    top_n=get_top_n(consine,N)
    

    return top_n


    #raise NotImplementedError()


# In[ ]:


def calculate_similarity(title,query):
    words_in_title=title.split()
    sum=0
    for token in query.split():
        if token in words_in_title:

            sum+=1
    return sum


# In[ ]:


def calculate_similarity_of_anchors(anchors,query):

    

    sum=0
    for id, text in anchors:
        words=text.split()
        #print(tokens)
        for token in query:
            if token in words:

                sum+=1
    return sum


# In[ ]:


def get_relevante_titles(index_title,query):
    titles=[]
    dic=index_title.id_to_title
    for id, title in dic.items():
        words_found=calculate_similarity(title,query)
        if words_found > 0:
            titles.append(((id,title),words_found))
    return titles
#{7: ['what', 'is', 'information', 'retrieval']}


# In[ ]:


def get_relevante_anchors(index_anchor,query):
    titles=[]
    dic=index_anchor.id_to_anchor

    for id, anchor in dic.items():
        words_found=calculate_similarity_of_anchors(anchor,query)
        if words_found > 0:
            titles.append(((id,anchor),words_found))
    return titles


# In[ ]:


def generate_graph(pages):
  ''' Compute the directed graph generated by wiki links.
  Parameters:
  -----------
    pages: RDD
      An RDD where each row consists of one wikipedia articles with 'id' and
      'anchor_text'.
  Returns:
  --------
    edges: RDD
      An RDD where each row represents an edge in the directed graph created by
      the wikipedia links. The first entry should the source page id and the
      second entry is the destination page id. No duplicates should be present.
    vertices: RDD
      An RDD where each row represents a vetrix (node) in the directed graph
      created by the wikipedia links. No duplicates should be present.
  '''
  # YOUR CODE HERE
  edges=[]
  vertices=[]
  lst=[]
  ls=[]
#pages.flatMap(lambda v: v[0])
  #edges=pages.map(lambda x: (x[0],k) for k,v in x[1])
 ## edges=pages.flatMap(x=>x)
  edges=pages.flatMap(lambda x: [Row(x[0], k) for k,v in x[1]]).distinct()
  #mylist=pages.mapValues(lambda v:)
  lst=pages.map(lambda x: Row(x[0]))
  ls=pages.flatMap(lambda x: [ Row(k) for k,v in x[1]]).distinct()
  vertices=lst.union(ls).distinct()
  return edges, vertices




def merge_results(title_scores,body_scores,title_weight=0.5,text_weight=0.5,N = 3):    
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body). 

    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows: 
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
                
    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows: 
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function. 
    
    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score). 
    """
    dic={}
    for key in title_scores.keys():
      dic[key]=[]
      firstlist=title_scores.get(key)
      secondlist=body_scores.get(key)
      #print(firstlist)
      #print(secondlist)
      for pair in firstlist:
        docid=pair[0]
        score1=pair[1]
        score2=0
        for pair2 in secondlist:
          if pair2[0]==docid:
            score2=pair2[1]
        mergedscore=(score1*title_weight)+(score2*text_weight)
        dic[key].append((docid,mergedscore))
      for pair in secondlist:
        docid=pair[0]
        score1=pair[1]
        
        found=False
        for pair2 in firstlist:
          if pair2[0]==docid:
            found=True
            
        
        if found==False:
          mergedscore=score1*text_weight
          dic[key].append((docid,mergedscore))
      dic[key]=list(set(dic.get(key)))
      dic[key]=sorted(dic.get(key),key=lambda x:x[1],reverse=True)
      dic[key]=dic.get(key)[:N]
     
   
    return dic
      


    # YOUR CODE HERE
    #raise NotImplementedError()






class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False





@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    bodysocres = get_topN_score_for_queries(query,index_body,100) #{q1:[(1,rank),(2,rank)]}

    titles=get_relevante_titles(index_title,query)
    titles=sorted(titles,key=lambda x:x[1],reverse=True)
    titlesscores=[]
    for pair in titles:
        titlesscores.append((pair[0][0],pair[1]))
    dic1={1:titlesscores}
    dic2={1:bodysocres}
    merged=merge_results(dic1,dic2,0.5,0.5,100)
    finalres=merged[1]
    finalList =[]
    wikiID = [tup[0] for tup in finalres]
    for id in wikiID:
        finalList.append((id,index_body.id_to_title[id]))
    res = finalList

    
   

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
   # print(query)
    cosSimDic = get_topN_score_for_queries(query,index_body,100) #{q1:[(1,rank),(2,rank)]}

    

    finalList =[]
    wikiID = [tup[0] for tup in cosSimDic]
    for id in wikiID:
        finalList.append((id,index_body.id_to_title[id]))
    res = finalList

    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        QUERY WORDS that appear in the title. For example, a document with a
        title that matches two of the query words will be ranked before a
        document with a title that matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    titles=get_relevante_titles(index_title,query)
    titles=sorted(titles,key=lambda x:x[1],reverse=True)
    for pair in titles:
        res.append(pair[0])
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    anchors=get_relevante_anchors(index_anchor, query)
    #print(anchors)
    anchors=sorted(anchors,key=lambda x:x[1],reverse=True)
    for pair in anchors:
        res.append((pair[0][0],index_body.id_to_title[pair[0][0]]))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    df =spark.read.csv(f'gs://{bucket_name}/pr/page_ranks.csv')
    for row in df.rdd.toLocalIterator():
        if int(row[0]) in wiki_ids:
            res.append((row[0],row[1]))
    res=sorted(res,key=lambda x: x[1],reverse=True)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    wid2pv = Counter()
    blob = bucket.blob('pageviews_202108_user.pkl')
    pickle_in = blob.download_as_string()
    wid2pv = pickle.loads(pickle_in)

    for val in wiki_ids:
        res.append((val,wid2pv[val]))
    res=sorted(res,key=lambda x:x[1],reverse=True)
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False)