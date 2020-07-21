import sys
import os
import tarfile

from userDefs import serialise, setupEssayConfig, getEssayParameter, stimestamp
from userDefs import loadFilesinPath, Stage1, Stage2
from nltk import SnowballStemmer
from nltk.corpus import stopwords
from pyspark import SparkContext

def main(essay_configs):

  # creates a Spark Context
  sc = SparkContext("local", "Simple App")

  # identifies all config files that comprises the essay
  if(os.path.isdir(essay_configs)):
    configFiles = [os.path.join(essay_configs, f) 
                    for f in os.listdir(essay_configs) 
                    if os.path.isfile(os.path.join(essay_configs, f))]
                       
  elif(os.path.isfile(essay_configs)):
    configFiles = [essay_configs]
    
  else:
    print "Parameter must be a file or directory: {0}".format(essay_configs)
    raise ValueError

  # run the pipeline as specified by each config file
  for configFile in configFiles:

    # loads essay configuration coded in the config file
    print setupEssayConfig(configFile)

    # recovers attributes related to essay automation
    scenario = getEssayParameter('EI_SCENARIO')
    essayid  = getEssayParameter('EI_ESSAYID')
    configid = getEssayParameter('EI_CONFIGID')

    # assures the essay slot (where files will be stored) is available
    runid = stimestamp()
    slot  = os.path.join('Essays', essayid, configid, runid)
    if(not os.path.exists(slot)): os.makedirs(slot)
    print 'Files are being stored at {0}'.format(slot)

    # loads training partition into an RDD
    # (S1i stands for  'Stage 1 input')
    baseDir = getEssayParameter('PO_BASEDIR')
    fraction = getEssayParameter('PO_FRACTION')
    inputPath = os.path.join(baseDir, '20news-bydate-train')
    (rddTrainS1i, rddTrainClass) = loadFilesinPath(sc, inputPath, fraction)

    # loads test partition into an RDD
    inputPath = os.path.join(baseDir, '20news-bydate-test')
    (rddTestS1i, rddTestClass) = loadFilesinPath(sc, inputPath, fraction)

    # applies pre-processing in both training and test RDDs
    rddTrainS2i = Stage1(rddTrainS1i)
    rddTestS2i = Stage1(rddTestS1i)

    # applies feature extraction in both training and test RDDs
    corpus = rddTrainS2i.union(rddTestS2i)
    rddTrainS3i = Stage2(sc, corpus, rddTrainS2i)
    rddTestS3i = Stage2(sc, corpus, rddTestS2i)

    # saves RDDs as lists in pickle file format
    
    # this RDD contains (docHID, original document path)
    serialise(rddTrainClass.collect(), os.path.join(slot, 'rddTrainClass'))
    
    # this RDD contains (docHID, original document path)
    serialise(rddTestClass.collect(), os.path.join(slot, 'rddTestClass'))
    
    # this RDD contains (docHID, {term:scores})
    serialise(rddTrainS3i.collect(), os.path.join(slot, 'rddTrainS3i'))     
    
    # this RDD contains (docHID, {term:scores})
    serialise(rddTestS3i.collect(), os.path.join(slot, 'rddTestS3i'))       

if __name__ == "__main__":
	
	if(not os.path.isdir()):
		tar = tarfile.open('20news-bydate.tar.gz')
		tar.extractall()
		tar.close()		
		
  main(sys.argv[1])