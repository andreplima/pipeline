import os
import re
import random
import cPickle as pickle
import ConfigParser

from math import log10
from datetime import datetime
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from collections import OrderedDict

#-------------------------------------------------------------------------------------------------------------------------------------------
# General Supporting definitions
#-------------------------------------------------------------------------------------------------------------------------------------------
# Capabilities: functions used throughout the code
#-------------------------------------------------------------------------------------------------------------------------------------------

def serialise(obj, name):
    f = open(name + '.pkl', 'wb')
    p = pickle.Pickler(f)
    p.fast = True
    p.dump(obj)
    f.close()
    p.clear_memo()

def deserialise(name):
    f = open(name + '.pkl', 'rb')
    p = pickle.Unpickler(f)
    obj = p.load()
    f.close()
    return obj

def timestamp():
    return(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

def stimestamp():
    return(datetime.now().strftime('%Y%m%d%H%M%S'))

#-------------------------------------------------------------------------------------------------------------------------------------------
# Essay Configuration definitions
#-------------------------------------------------------------------------------------------------------------------------------------------
# Capabilities: functions related to parameter files interface
#-------------------------------------------------------------------------------------------------------------------------------------------

# Essay Parameters hashtable
EssayParameters = {}

def setupEssayConfig(configFile = ''):

    # set parameters related to essay identification
    setEssayParameter('EI_SCENARIO', 'None')
    setEssayParameter('EI_ESSAYID',  'None')
    setEssayParameter('EI_CONFIGID', 'None')

    # set parameters related to pipeline options
    setEssayParameter('PO_BASEDIR',            './Instances')
    setEssayParameter('PO_PARTITIONS',         '2')
    setEssayParameter('PO_FRACTION',           '1.00')
    setEssayParameter('PO_SEED',               '23')
    setEssayParameter('PO_TOKENIZING',         '0')
    setEssayParameter('PO_LOWERCASE',          '0')
    setEssayParameter('PO_STOPWORDS',          '[]')
    setEssayParameter('PO_LANGUAGE',           '0')
    setEssayParameter('PO_STEMMING',           '0')
    setEssayParameter('PO_FEATURE_EXTRACTION', '0')

    # overrides default values with user-defined configuration
    loadEssayConfig(configFile)

    # initialises the random number generator
    random.seed(getEssayParameter('PO_SEED'))

    return listEssayConfig()


# sets the value of a specific parameter
# when using inside python code, declare value as string, independently of its true type. example: 'True', '0.32', 'Rastrigin, only Reproduction'
# when using parameters in Config files, declare value as if it was a string, except of the use of ''. example: True, 0.32, Rastrigin, only Reproduction
def setEssayParameter(param, value):

    PO_param = param.upper()

    # boolean-valued parameters
    if(PO_param in [None]):

        PO_value = eval(value[0]) if isinstance(value, list) else bool(value)

    # integer-valued parameters
    elif(PO_param in ['PO_PARTITIONS', 'PO_SEED', 'PO_TOKENIZING', 'PO_LOWERCASE',  'PO_LANGUAGE',  'PO_STEMMING', 'PO_FEATURE_EXTRACTION']):

        PO_value = eval(value[0])

    # floating-point-valued parameters
    elif(PO_param in ['PO_FRACTION']):

        PO_value = float(eval(value[0]))

    # parameters that requires eval expansion
    elif(PO_param in ['PO_STOPWORDS']):

        PO_value = value

    # parameters that represent text
    else:

        PO_value = value[0]
        #PO_value = value

    EssayParameters[PO_param] = PO_value


def getEssayParameter(param):

    return EssayParameters[param.upper()]


class OrderedMultisetDict(OrderedDict):
    def __setitem__(self, key, value):

        try:
            item = self.__getitem__(key)
        except KeyError:
            super(OrderedMultisetDict, self).__setitem__(key, value)
            return

        if isinstance(value, list):
            item.extend(value)
        else:
            item.append(value)
        super(OrderedMultisetDict, self).__setitem__(key, item)


# loads essay configuration coded in a essay parameters file
def loadEssayConfig(configFile):

    if(len(configFile) > 0):

        if(os.path.exists(configFile)):

            # initialises the config parser and set a custom dictionary in order to allow multiple entries
            # of a same key (example: several instances of GA_ESSAY_ALLELE
            config = ConfigParser.RawConfigParser(dict_type = OrderedMultisetDict)
            config.read(configFile)

            # loads parameters coded in the Essay section
            for param in config.options('Essay'):
                setEssayParameter(param, config.get('Essay', param))

            # loads parameters coded in the Pipeline section
            for param in config.options('Pipeline'):
                setEssayParameter(param, config.get('Pipeline', param))

            # expands parameter values that requires evaluation
            EssayParameters['PO_STOPWORDS'] = eval(EssayParameters['PO_STOPWORDS'][0])

        else:

            print "[{0}] Warning: Configuration file [{1}] was not found".format(timestamp(), configFile)



# recovers the current essay configuration
def listEssayConfig():

    res = ''
    for e in sorted(EssayParameters.items()):
        res = res + "{0} : {1} (as {2})\n".format(e[0], e[1], type(e[1]))

    return res


#-------------------------------------------------------------------------------------------------------------------------------------------
# Pipeline supporting definitions
#-------------------------------------------------------------------------------------------------------------------------------------------
# Capabilities: functions related to preprocessing textual content
#-------------------------------------------------------------------------------------------------------------------------------------------

def loadFilesinPath(sc, inputPath, fraction):

  numOfPartitions = getEssayParameter('PO_PARTITIONS')
  if os.path.exists(inputPath):

    rdd1 = sc.parallelize([], numOfPartitions)
    for x in os.walk(inputPath):
      rddt = (sc.wholeTextFiles(x[0], numOfPartitions))
      if(fraction < 1.0):
        randomSeed = getEssayParameter('PO_SEED')
        rddt = rddt.sample(False, fraction, randomSeed)
      rdd1 = rdd1.union(rddt)

    rdd = rdd1.map(lambda (path, content): (hash(path), content))
    rddH2F = rdd1.map(lambda (path, content): (hash(path), path))

  else:

    rdd = sc.parallelize([], numOfPartitions)
    rddH2F = sc.parallelize([], numOfPartitions)

  return (rdd, rddH2F)

def foldCase(documentContent, parameter):
  if(parameter == 1):
    documentContent = documentContent.lower()
  return documentContent

def tokenize(documentContent, parameter):

  if(parameter == 0):
    # specifies tokens comprised of Latin letters
    regExpr = '[^a-zA-Z]'

  elif(parameter == 1):
    # (same segmentation employed by PreTexT)
    documentContent = documentContent.replace("_"," ")
    regExpr = '\W+'

  else:
    regExpr = ''

  return filter(None, re.split(regExpr, documentContent))

def removeStopWords(listOfTokens, listOfStopWords):
  return [token for token in listOfTokens if token not in listOfStopWords]

def applyStemming(listOfTokens, stemmer):
  return [stemmer.stem(token) for token in listOfTokens]

def tf(listOfTokens, normalise = True):

  # creates a dictionary relating each token
  # to the number of times it occurs in its source document
  types = {}
  for token in listOfTokens:
    if(token in types.keys()):
        types[token] = types[token] + 1
    else:
        types[token] = 1

  # normalises the counts using the total number of tokens
  # found in the source document
  if(normalise):
    nTokens = len(listOfTokens)
    for term in types.keys():
        types[term] =types[term]/float(nTokens)

  return types

def idf_dict(corpus):

  # creates an RDD where each object represents a list of distinct terms
  # occurring in a specific document
  terms = corpus.map(lambda (docHID, listOfTokens): list(set(listOfTokens)))

  # creates an RDD where each object represents the number of documents
  # where a specific term occurs
  term1Tuples = (terms.flatMap(lambda listOfTerms:
                                      [(term, 1) for term in listOfTerms] ))

  termCounts = term1Tuples.reduceByKey(lambda a, b: a + b)

  # creates an RDD where each object represents a term and its idf score
  nDocuments = corpus.count()
  rddIdf = (termCounts.map(lambda (term, nHits):
                                  (term, log10(nDocuments/float(nHits))) ))

  return (rddIdf)

def idf(listOfTokens, dictIdf):
  listOfIdfScoredTokens = {term: dictIdf[token] for token in listOfTokens}
  return listOfIdfScoredTokens

def tfidf(listOfTokens, dictIdf, normalise = True):

  listOfTfScoredTokens = tf(listOfTokens, normalise)
  listOfTfidfScoredTokens = ({token: listOfTfScoredTokens[token] * dictIdf[token]
                                     for token in listOfTfScoredTokens})

  return listOfTfidfScoredTokens

def Stage1(rddS1i):

  # identifies the selected language
  if(getEssayParameter('PO_LANGUAGE') == 0):
    textLang = 'english'
  elif(getEssayParameter('PO_LANGUAGE') == 1):
    textLang = 'portuguese'

  # initialises a stemmer and the list of stop words
  stemmer = SnowballStemmer(textLang)
  listOfStopWords = getEssayParameter('PO_STOPWORDS')
  if(listOfStopWords == ['default']):
    # if user requests the default list, recover the one offered by NLTK
    listOfStopWords = stopwords.words(textLang)

  # Step 1 - applies case folding
  po_lowercase = getEssayParameter('PO_LOWERCASE')
  if(po_lowercase > 0):
    rddS1A1 = (rddS1i
               .map(lambda (docHID, content):
                           (docHID, foldCase(content, po_lowercase)) ))
  else:
    rddS1A1 = rddS1i

  # Step 2 - tokenises document content
  po_tokenizing = getEssayParameter('PO_TOKENIZING')
  rddS1A2 = (rddS1A1
             .map(lambda (docHID, content):
                         (docHID, tokenize(content, po_tokenizing)) ))

  # Step 3 - removes stop words
  po_stopWords = getEssayParameter('PO_STOPWORDS')
  if(po_stopWords > 0):
    rddS1A3 = (rddS1A2
               .map(lambda (docHID, listOfTokens):
                           (docHID, removeStopWords(listOfTokens, listOfStopWords))
                    ))
  else:
    rddS1A3 = rddS1A2

  # Step 4 - applies stemming
  po_stemming = getEssayParameter('PO_STEMMING')
  if(po_stemming > 0):
    rddS1A4 = (rddS1A3
               .map(lambda (docHID, listOfTokens):
                           (docHID, applyStemming(listOfTokens, stemmer)) ))
  else:
    rddS1A4 = rddS1A3

  return rddS1A4

def Stage2(sc, corpusS2i, rddS2i):

  po_feature_extraction = getEssayParameter('PO_FEATURE_EXTRACTION')

  # option 0 - computes only term frequency (unnormalised)
  # option 1 - computes only term frequency (normalised)
  # option 2 - computes only idf scores
  # option 3 - computes tfidf scores (using unnormalised term frequency)
  # option 4 - computes tfidf scores (using normalised term frequency)

  # computes tf for each document/term
  if(po_feature_extraction == 0):
    rddS2A1 = (rddS2i
               .map(lambda (docID, tokens):
                           (docID, tf(tokens, False)) ))

  # computes normalised tf for each document/term
  elif(po_feature_extraction == 1):
    rddS2A1 = (rddS2i
               .map(lambda (docID, tokens):
                           (docID, tf(tokens, True)) ))

  # computes idf scores for each document/term
  elif(po_feature_extraction == 2):

    # creates an RDD with IDF scores for all terms in corpus
    dictIdf = idf_dict(corpusS2i).collectAsMap()

    # broadcasts a copy of the idf dictionary to worker processes
    bcDictIdf = sc.broadcast(dictIdf)

    # computes the idf scores for each term
    rddS2A1 = (rddS2i
               .map(lambda (docID, tokens):
                           (docID, idf(tokens, bcDictIdf.value)) ))

  # computes tf-idf scores for each document/term
  elif(po_feature_extraction == 3):

    # creates an RDD with IDF scores for all terms in corpus
    dictIdf = idf_dict(corpusS2i).collectAsMap()

    # broadcasts a copy of the idf dictionary to worker processes
    bcDictIdf = sc.broadcast(dictIdf)

    # computes the tfidf scores for each term
    rddS2A1 = (rddS2i
               .map(lambda (docID, tokens):
                           (docID, tfidf(tokens, bcDictIdf.value, False)) ))

  # computes normalised tf-idf scores for each document/term
  elif(po_feature_extraction == 4):

    # creates an RDD with IDF scores for all terms in corpus
    dictIdf = idf_dict(corpusS2i).collectAsMap()

    # broadcasts a copy of the idf dictionary to worker processes
    bcDictIdf = sc.broadcast(dictIdf)

    # computes the tfidf scores for each term
    rddS2A1 = (rddS2i
               .map(lambda (docID, tokens):
                           (docID, tfidf(tokens, bcDictIdf.value, True)) ))

  return rddS2A1

# a and b must conform to {term:score}
def dotprod(a, b):
  return sum([a[i] * b[j] for i in a.keys() for j in b.keys() if i == j])

# a must conform to {term:score}
def norm(a):
  return (dotprod(a,a) ** 0.5)

# calculates cosine similarity
# a and b must conform to {term:score}
def cossim(a, b):
  return (dotprod(a,b) / (norm(a) * norm(b)))


