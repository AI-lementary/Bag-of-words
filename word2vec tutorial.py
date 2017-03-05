import pandas as pd
#Read data from files
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)
#verify the number of reviews that were read
print ("Read %d labeled train reviews, %d labeled test reviews," \
       "and %d unlabeled reviews\n" % (train['review'].size, test['review'].size,unlabeled_train['review'].size))

#Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist(review, remove_stopwords=True):
    #Function to convert a dcoument to a sequence of words,
    #optionally removing stopwords
    #
    #1. remove html
    review_text = BeautifulSoup(review,"lxml").get_text()
    
    #2. remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    
    #3. Convert words to lower case and split them
    words = review_text.lower().split()
    
    #4. optionally remove stopwords (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #5. Return a list of words
    return(words)

print(train.loc[0]['review'])
print(review_to_wordlist(train.loc[0]['review']))

        