import os
import re
import cPickle as pickle
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from math import log10

def loadCorpus(sourcepath):

  corpus = {}
  for filename in os.listdir(sourcepath):
    print('Carregando arquivo {0}'.format(filename))
    fh = open(os.path.join(sourcepath, filename), 'r')
    corpus[filename] = fh.readlines()
    fh.close()

  return corpus

def processCorpus(corpus, params):

  # recupera os parametros
  (param_foldCase, param_tokenize, param_listOfStopWords, param_stemmer) = params

  newCorpus = {}
  for document in corpus:
    content = []
    for sentence in corpus[document]:
      sentence = sentence.rstrip('\n')
      sentence = foldCase(sentence, param_foldCase)
      listOfTokens = tokenize(sentence)
      listOfTokens = removeStopWords(listOfTokens, param_listOfStopWords)
      listOfTokens = applyStemming(listOfTokens, param_stemmer)
      content.append(listOfTokens)

    newCorpus[document] = content

  return newCorpus

def representCorpus(corpus):

  # cria um dicionario que associa um documento com a lista de tokens que ocorrem nele
  # (a divisao por sentencas eh removida e um termo pode ocorrer multiplas vezes num documento)
  newCorpus = {}
  for document in corpus:
    newCorpus[document] = [token for sentence in corpus[document] for token in sentence]

  # cria uma lista com todos os tokens que ocorrem em cada documento
  # se um token ocorre mais do que uma vez em um documento, apenas uma ocorrencia eh registrada
  allTokens = []
  for document in newCorpus:
    allTokens = allTokens + list(set(newCorpus[document]))

  # cria o dicionario reverso, indicando para cada token o numero de documentos no qual ele ocorre
  idfDict = {}
  for token in allTokens:
    try:
      idfDict[token] += 1
    except KeyError:
      idfDict[token] = 1

  # atualiza o dicionario reverso, associando cada token com seu idf score
  nDocuments = len(corpus)
  for token in idfDict:
    idfDict[token] = log10(nDocuments/float(idfDict[token]))

  # computa a representacao vetorial de cada documento, na forma de vetores com tf-idf scores
  for document in newCorpus:

    # computa um dicionario com os tf scores de cada termo que ocorre no documento
    dictOfTfScoredTokens = tf(newCorpus[document])
    dictOfTfidfScoredTokens = ({token: dictOfTfScoredTokens[token] * idfDict[token] for token in dictOfTfScoredTokens})
    newCorpus[document] = dictOfTfidfScoredTokens

  return newCorpus

def foldCase(sentence, parameter):
  # reduz o caixa dos caracteres se parameter == True
  if(parameter): sentence = sentence.lower()
  return sentence

def tokenize(sentence):
  # gera tokens iguais aos gerados pela ferramenta PreTexT
  sentence = sentence.replace("_"," ")
  regExpr = '\W+'
  return filter(None, re.split(regExpr, sentence))

def removeStopWords(listOfTokens, listOfStopWords):
  return [token for token in listOfTokens if token not in listOfStopWords]

def applyStemming(listOfTokens, stemmer):
  return [stemmer.stem(token) for token in listOfTokens]

def tf(listOfTokens):

  # cria um dicionario associando cada token com o numero de vezes
  # em que ele ocorre no documento (cujo conteudo eh listOfTokens)
  types = {}
  for token in listOfTokens:
    if(token in types.keys()):
        types[token] = types[token] + 1
    else:
        types[token] = 1

  return types

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


def main():

  # estagio de preparacao
  sourcepath = os.path.join('.', 'corpus')
  corpus1 = loadCorpus(sourcepath)

  # estagio de pre-processamemento
  param_foldCase = True
  param_tokenize = True
  param_idiom = 'english'
  param_listOfStopWords = stopwords.words(param_idiom)
  param_stemmer = SnowballStemmer(param_idiom)
  params = (param_foldCase, param_tokenize, param_listOfStopWords, param_stemmer)

  corpus2 = processCorpus(corpus1, params)

  # estagio de representacao
  targetpath = os.path.join('.')
  corpus3 = representCorpus(corpus2)
  serialise(corpus3, os.path.join(targetpath, 'corpus'))

if __name__ == '__main__':
  main()
